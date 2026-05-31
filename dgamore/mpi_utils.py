# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import mpi4py.MPI as MPI
import numpy as np
import scipy.fft as fft

import dgamore.config as config
from dgamore import symmetry_reduction
from dgamore.brillouin_zone import KGrid
from dgamore.four_point import FourPoint
from dgamore.mpi_distributor import MpiDistributor, MAX_MPI_BYTES


def _get_node_aware_v_dist(n_nu, comm):
    """
    Calculates frequency distribution based on physical node topology.
    Frequencies are split equally amongst nodes, then assigned to ranks within those nodes.

    Uses inverse rank-lookup to be robust against inconsistent hostname strings
    returned by the OS in local or cluster environments.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Group ranks by physical hostname
    local_hostname = str(MPI.Get_processor_name()).strip()
    all_hostnames = comm.allgather(local_hostname)

    # Map hostnames to the list of ranks living on them
    nodes_map = {}
    for r, h in enumerate(all_hostnames):
        h_clean = str(h).strip()
        if h_clean not in nodes_map:
            nodes_map[h_clean] = []
        nodes_map[h_clean].append(r)

    # Sorted list of unique nodes ensures every rank sees the same order
    sorted_node_names = sorted(nodes_map.keys())
    n_nodes = len(sorted_node_names)

    # 2. CANONICAL HOSTNAME RESOLUTION
    # Instead of string matching, find which node list contains THIS rank.
    # This guarantees we find the correct key in nodes_map.
    canonical_hostname = None
    for name, ranks in nodes_map.items():
        if rank in ranks:
            canonical_hostname = name
            break

    if canonical_hostname is None:
        raise RuntimeError(f"Rank {rank} could not find itself in the host map.")

    # 3. Distribute frequencies to nodes
    v_per_node = n_nu // n_nodes
    extra_v_nodes = n_nu % n_nodes

    my_node_idx = sorted_node_names.index(canonical_hostname)
    v_on_this_node = v_per_node + (1 if my_node_idx < extra_v_nodes else 0)

    # 4. Split this node's frequencies amongst its local ranks
    ranks_on_my_node = nodes_map[canonical_hostname]
    rank_in_node = ranks_on_my_node.index(rank)

    v_per_rank = v_on_this_node // len(ranks_on_my_node)
    extra_v_ranks = v_on_this_node % len(ranks_on_my_node)

    my_size = v_per_rank + (1 if rank_in_node < extra_v_ranks else 0)

    # 5. Globalize the distribution for the Distributor
    # Allgather ensures every rank knows the frequency slices of every other rank.
    all_sizes = np.zeros(size, dtype=int)
    all_sizes[rank] = my_size
    comm.Allgather(MPI.IN_PLACE, all_sizes)

    # Calculate slices based on the gathered sizes
    all_offsets = np.insert(np.cumsum(all_sizes), 0, 0)
    all_slices = [slice(all_offsets[i], all_offsets[i + 1]) for i in range(size)]

    return all_sizes, all_slices


def _send_in_chunks(comm, arr, dest, base_tag=0):
    """
    Sends a numpy array in chunks to a destination rank without a handshake.
    """
    arr = np.ascontiguousarray(arr)
    itemsize = arr.dtype.itemsize
    items_per_q = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1

    # Calculate how many q-rows fit into the 2GB limit
    max_q_per_chunk = max(1, MAX_MPI_BYTES // (items_per_q * itemsize))
    total_q = arr.shape[0]

    for i in range(0, total_q, max_q_per_chunk):
        j = min(total_q, i + max_q_per_chunk)
        # Use the loop index as a tag offset to keep chunks ordered
        comm.Send(arr[i:j], dest=dest, tag=base_tag + (i // max_q_per_chunk))


def _recv_in_chunks(comm, shape, dtype, source, base_tag=0):
    """
    Receives a numpy array in chunks from a source rank into a pre-allocated buffer.
    """
    out = np.empty(shape, dtype=dtype)
    itemsize = np.dtype(dtype).itemsize
    items_per_q = int(np.prod(shape[1:])) if len(shape) > 1 else 1

    max_q_per_chunk = max(1, MAX_MPI_BYTES // (items_per_q * itemsize))
    total_q = shape[0]

    for i in range(0, total_q, max_q_per_chunk):
        j = min(total_q, i + max_q_per_chunk)
        # Receive directly into the slice of the output buffer
        comm.Recv(out[i:j], source=source, tag=base_tag + (i // max_q_per_chunk))

    return out


def map_irrbz_fullbz(obj, mpi_dist_irrk, mpi_dist_fullbz):
    """
    Regular way of mapping an object from the irreducible BZ to the full BZ. Processes all data on a single rank.
    """
    obj.mat = mpi_dist_irrk.gather(obj.mat)
    if mpi_dist_irrk.comm.rank == 0:
        obj = obj.map_to_full_bz(config.lattice.q_grid)
    obj.mat = mpi_dist_fullbz.scatter(obj.mat)
    return obj


def exchange_and_map_irrbz_fullbz(
    obj: FourPoint, mpi_dist_irrk: MpiDistributor, mpi_dist_fullbz: MpiDistributor
) -> FourPoint:
    """
    Maps an obj from the irreducible BZ distribution to the full BZ distribution without
    ever assembling the full object on any single rank.

    Each rank holds a slice of the obj over the irreducible BZ (shape [q_irr_rank, ...]).
    This routine redistributes the data peer-to-peer so that each rank ends up with a slice
    over the full BZ (shape [q_full_rank, ...]), with symmetry-equivalent points correctly
    replicated according to the irrk_inv mapping.

    If ``config.lattice.q_grid`` is in auto-discovered symmetry mode (its
    ``specify_auto_symmetries`` has been called), the per-k orbital transformation
    ``(sigma_k, U_k, conj_k)`` is also applied locally on each rank, using only the
    transformation arrays sliced to that rank's FBZ range. No global gather is needed.

    This is a distributed replacement for the pattern (see also 'map_irrbz_fullbz')

        obj.mat = mpi_dist_irrk.gather(obj.mat)
        if comm.rank == 0:
            obj = obj.map_to_full_bz(q_grid)
        obj.mat = mpi_dist_fullbz.scatter(obj.mat)

    which would require rank 0 to hold the entire full-BZ obj in memory.
    """
    comm = mpi_dist_irrk.comm
    rank = comm.rank
    size = comm.size

    q_grid = config.lattice.q_grid

    # 1. Global mapping setup
    # irrk_inv[fbz_idx] = irrbz_idx
    irrk_inv_flat = q_grid.irrk_inv.ravel()

    # These are the global FBZ indices this specific rank is responsible for
    my_fbz_range = np.arange(mpi_dist_fullbz.my_slice.start, mpi_dist_fullbz.my_slice.stop)
    # These are the corresponding global IRBZ indices needed
    needed_irrk_indices = irrk_inv_flat[my_fbz_range]

    # 2. Identify Sources
    # Find which rank owns each needed IRBZ index
    irr_rank_starts = np.array([s.start for s in mpi_dist_irrk.slices])
    # owner_ranks[i] is the rank that has the data for my_fbz_range[i]
    owner_ranks = np.searchsorted(irr_rank_starts, needed_irrk_indices, side="right") - 1

    # 3. Request/Send Index Information
    # We need to tell each rank exactly which of its LOCAL indices we need.
    # To be efficient, we only ask for each unique index once.

    indices_to_send = [[] for _ in range(size)]
    # Mapping to help us rebuild the full_mat after receiving unique matrices
    # key: source_rank, value: (unique_local_indices, map_to_my_fbz_slice)
    receiving_info = {}

    for src in range(size):
        mask = owner_ranks == src
        if not np.any(mask):
            continue

        global_indices = needed_irrk_indices[mask]
        local_indices = global_indices - irr_rank_starts[src]

        # Uniqueify so we don't transfer the same matrix multiple times
        unique_local, inv_map = np.unique(local_indices, return_inverse=True)
        indices_to_send[src] = unique_local.astype(int)

        # Store how to put these unique received matrices back into our full_mat
        receiving_info[src] = {
            "full_mat_locations": np.where(mask)[0],
            "unique_map": inv_map,
            "count": unique_local.size,
        }

    # 4. Exchange Counts and Indices
    send_counts = np.array([len(indices_to_send[r]) for r in range(size)], dtype=int)
    recv_counts = np.empty(size, dtype=int)
    comm.Alltoall(send_counts, recv_counts)

    reqs = []
    remote_indices_needed_from_me = [np.empty(recv_counts[r], dtype=int) for r in range(size)]

    for r in range(size):
        if r == rank:
            continue
        if send_counts[r] > 0:
            reqs.append(comm.Isend(indices_to_send[r], dest=r, tag=11))
        if recv_counts[r] > 0:
            reqs.append(comm.Irecv(remote_indices_needed_from_me[r], source=r, tag=11))
    MPI.Request.Waitall(reqs)

    # 5. Data Exchange
    # Prepare buffers
    rest_shape = obj.mat.shape[1:]
    dtype = obj.mat.dtype
    full_mat = np.empty((mpi_dist_fullbz.my_size,) + rest_shape, dtype=dtype)

    data_reqs = []
    send_buffers = []  # Keep alive for Isend

    # Handle Self-Copy first (Avoids MPI latency for local data)
    if rank in receiving_info:
        info = receiving_info[rank]
        local_data = obj.mat[indices_to_send[rank]]
        full_mat[info["full_mat_locations"]] = local_data[info["unique_map"]]

    # Prepare Receives
    receive_buffers = {}
    for src, info in receiving_info.items():
        if src == rank:
            continue
        buf = np.empty((info["count"],) + rest_shape, dtype=dtype)
        receive_buffers[src] = buf
        data_reqs.append(comm.Irecv(buf, source=src, tag=12))

    # Prepare Sends
    for dest in range(size):
        if dest == rank or recv_counts[dest] == 0:
            continue
        # Extract the matrices the remote rank requested from my local IRBZ slice
        data_to_send = np.ascontiguousarray(obj.mat[remote_indices_needed_from_me[dest]])
        send_buffers.append(data_to_send)
        data_reqs.append(comm.Isend(data_to_send, dest=dest, tag=12))

    MPI.Request.Waitall(data_reqs)

    # 6. Reconstruct full_mat from received unique buffers
    for src, buf in receive_buffers.items():
        info = receiving_info[src]
        # Duplicate the unique received matrices into their (multiple) FBZ target rows
        full_mat[info["full_mat_locations"]] = buf[info["unique_map"]]

    # 6b. Apply per-k orbital transformation locally if q_grid is in auto mode.
    # Each rank only needs the FBZ slice of (Us, sigmas, conjs) corresponding to
    # its own my_fbz_range. No gather, no scatter.
    if getattr(q_grid, "is_auto", False) and mpi_dist_fullbz.my_size > 0:
        # FourPoint always has 4 orbital indices contracted with the symmetry.
        num_orb_dims = 4
        nb_full = q_grid._auto_us.shape[-1]
        us_flat = q_grid._auto_us.reshape(-1, nb_full, nb_full)
        sigmas_flat = q_grid._auto_sigmas.reshape(-1)
        conjs_flat = q_grid._auto_conjs.reshape(-1)

        us_local = us_flat[my_fbz_range]
        sigmas_local = sigmas_flat[my_fbz_range]
        conjs_local = conjs_flat[my_fbz_range]

        full_mat = symmetry_reduction.apply_auto_orbital_transform(
            full_mat,
            us=us_local,
            sigmas=sigmas_local,
            conjs=conjs_local,
            num_orbital_dimensions=num_orb_dims,
        )

    # 7. Finalize Object
    return FourPoint(
        full_mat,
        obj.channel,
        obj.nq,
        obj.num_wn_dimensions,
        obj.num_vn_dimensions,
        obj.full_niw_range,
        obj.full_niv_range,
        True,
        obj.frequency_notation,
    )


def gather_full_ibz_for_vslice(
    gamma_r: FourPoint, mpi_dist_irrq: MpiDistributor, mpi_dist_v: MpiDistributor, q_grid: KGrid
) -> FourPoint:
    # 1. Distribution Update (Node-Aware)
    sizes, slices = _get_node_aware_v_dist(mpi_dist_v.ntasks, mpi_dist_v.comm)
    mpi_dist_v._sizes, mpi_dist_v._slices = sizes, slices
    mpi_dist_v._my_size = sizes[mpi_dist_v.my_rank]

    comm = mpi_dist_irrq.comm
    rank = mpi_dist_irrq.my_rank
    size = mpi_dist_irrq.mpi_size
    dtype = gamma_r.mat.dtype

    orb_dims = gamma_r.mat.shape[1:-2]
    n_vp = gamma_r.mat.shape[-1]
    items_per_q_v = int(np.prod(orb_dims)) * mpi_dist_v.my_size * n_vp

    # 2. Pre-allocate Buffer
    if mpi_dist_v.my_size > 0:
        full_ibz_mat = np.zeros((mpi_dist_irrq.ntasks,) + orb_dims + (mpi_dist_v.my_size, n_vp), dtype=dtype)
    else:
        full_ibz_mat = None

    # 3. Non-Blocking Exchange (The "Fast" Way)
    reqs = []
    send_buffers = []  # Protect from Garbage Collection

    # A. Pre-post Receives (Matches the exchange_and_map logic)
    if mpi_dist_v.my_size > 0:
        for r_src in range(size):
            q_src_count = mpi_dist_irrq.sizes[r_src]
            if q_src_count == 0:
                continue

            # Using the same chunking logic as your exchange method
            max_q_recv = max(1, MAX_MPI_BYTES // (items_per_q_v * dtype.itemsize))
            q_offset = mpi_dist_irrq.slices[r_src].start

            for chunk_idx, i in enumerate(range(0, q_src_count, max_q_recv)):
                j = min(q_src_count, i + max_q_recv)
                tag = (r_src * size + rank) + chunk_idx
                reqs.append(comm.Irecv(full_ibz_mat[q_offset + i : q_offset + j], source=r_src, tag=tag))

    # B. Post Sends
    for r_dst in range(size):
        v_dst_size = mpi_dist_v.sizes[r_dst]
        if v_dst_size == 0 or mpi_dist_irrq.my_size == 0:
            continue

        v_dst_slice = mpi_dist_v.slices[r_dst]
        items_per_q_send = int(np.prod(orb_dims)) * v_dst_size * n_vp
        max_q_send = max(1, MAX_MPI_BYTES // (items_per_q_send * dtype.itemsize))

        for chunk_idx, i in enumerate(range(0, mpi_dist_irrq.my_size, max_q_send)):
            j = min(mpi_dist_irrq.my_size, i + max_q_send)
            tag = (rank * size + r_dst) + chunk_idx

            # Payload must be contiguous for Send
            payload = np.ascontiguousarray(gamma_r.mat[i:j, ..., v_dst_slice, :])
            send_buffers.append(payload)
            reqs.append(comm.Isend(payload, dest=r_dst, tag=tag))

    # 4. Wait for All to complete
    MPI.Request.Waitall(reqs)

    # 5. Local Expansion
    if mpi_dist_v.my_size > 0:
        gamma_r.mat = full_ibz_mat
        return gamma_r.map_to_full_bz(q_grid)
    else:
        return None


def get_pencil_indices(rank: int, size: int, nq: tuple[int, int, int], layout: str) -> np.ndarray:
    """
    Calculates which global q-indices (0 to n_tot-1) a rank owns
    based on the decomposition layout.

    nq: tuple (nx, ny, nz)
    layout: 'flat', 'z_pencil', 'y_pencil', 'x_pencil'
    """
    nx, ny, nz = nq
    n_tot = nx * ny * nz

    if layout == "flat":
        # Same convention as MpiDistributor._distribute_tasks: excess on the LAST ranks.
        n_per, rem = divmod(n_tot, size)
        sizes = np.full(size, n_per, dtype=int)
        if rem:
            sizes[-rem:] += 1
        start = int(sizes[:rank].sum())
        count = int(sizes[rank])
        return np.arange(start, start + count)
    elif layout == "z_pencil":
        # A Z-pencil owns all nz points for a specific (x, y) coordinate.
        # Total number of such pencils is nx * ny.
        n_pencils = nx * ny
        n_per, rem = divmod(n_pencils, size)
        start_p = rank * n_per + min(rank, rem)
        count_p = n_per + (1 if rank < rem else 0)

        # In a flattened array [x,y,z], a Z-pencil is a contiguous block of length nz.
        # The global start index of pencil 'p' is p * nz.
        indices = []
        for p in range(start_p, start_p + count_p):
            indices.append(np.arange(p * nz, (p + 1) * nz))
        return np.concatenate(indices) if indices else np.array([], dtype=int)
    elif layout == "y_pencil":
        # A Y-pencil owns all ny points for a specific (x, z) coordinate.
        # Total number of such pencils is nx * nz.
        n_pencils = nx * nz
        n_per, rem = divmod(n_pencils, size)
        start_p = rank * n_per + min(rank, rem)
        count_p = n_per + (1 if rank < rem else 0)

        indices = []
        for p in range(start_p, start_p + count_p):
            # Decompose pencil index p into x and z
            ix = p // nz
            iz = p % nz
            # A Y-pencil starts at (ix, 0, iz) and jumps by nz for ny steps.
            # Global index q = ix*(ny*nz) + iy*nz + iz
            start_q = ix * (ny * nz) + iz
            indices.append(start_q + np.arange(ny) * nz)
        return np.concatenate(indices) if indices else np.array([], dtype=int)
    elif layout == "x_pencil":
        # An X-pencil owns all nx points for a specific (y, z) coordinate.
        # Total number of such pencils is ny * nz.
        n_pencils = ny * nz
        n_per, rem = divmod(n_pencils, size)
        start_p = rank * n_per + min(rank, rem)
        count_p = n_per + (1 if rank < rem else 0)

        indices = []
        for p in range(start_p, start_p + count_p):
            # p represents the (y, z) coordinate
            iy = p // nz
            iz = p % nz
            # An X-pencil starts at (0, iy, iz) and jumps by (ny*nz) for nx steps.
            # Global index q = ix*(ny*nz) + iy*nz + iz
            start_q = iy * nz + iz
            indices.append(start_q + np.arange(nx) * (ny * nz))
        return np.concatenate(indices) if indices else np.array([], dtype=int)
    else:
        raise ValueError(f"Unknown layout: {layout}")


def _redistribute_p2p(mat, nq, comm, source_layout, target_layout):
    size = comm.Get_size()
    rank = comm.Get_rank()

    src_indices = get_pencil_indices(rank, size, nq, source_layout)
    tgt_indices = get_pencil_indices(rank, size, nq, target_layout)

    res_mat = np.empty((len(tgt_indices),) + mat.shape[1:], dtype=mat.dtype)
    src_map = {g_idx: l_idx for l_idx, g_idx in enumerate(src_indices)}
    tgt_map = {g_idx: l_idx for l_idx, g_idx in enumerate(tgt_indices)}

    for shift in range(size):
        target_rank = (rank + shift) % size
        source_rank = (rank - shift) % size

        remote_tgt_indices = get_pencil_indices(target_rank, size, nq, target_layout)
        to_send_g = np.intersect1d(src_indices, remote_tgt_indices, assume_unique=True)

        remote_src_indices = get_pencil_indices(source_rank, size, nq, source_layout)
        to_recv_g = np.intersect1d(tgt_indices, remote_src_indices, assume_unique=True)

        reqs = []
        send_buf = None  # keep alive until Waitall
        recv_staging = None

        if len(to_send_g) > 0:
            send_l = [src_map[g] for g in to_send_g]
            send_buf = np.ascontiguousarray(mat[send_l])
            send_view = send_buf.view(np.byte)
            for i in range(0, send_view.nbytes, MAX_MPI_BYTES):
                reqs.append(comm.Isend(send_view[i : i + MAX_MPI_BYTES], dest=target_rank, tag=shift))

        if len(to_recv_g) > 0:
            recv_staging = np.empty((len(to_recv_g),) + mat.shape[1:], dtype=mat.dtype)
            recv_view = recv_staging.view(np.byte)
            for i in range(0, recv_view.nbytes, MAX_MPI_BYTES):
                reqs.append(comm.Irecv(recv_view[i : i + MAX_MPI_BYTES], source=source_rank, tag=shift))

        MPI.Request.Waitall(reqs)

        # Now copy from staging into res_mat at the right rows
        if len(to_recv_g) > 0:
            recv_l = [tgt_map[g] for g in to_recv_g]
            res_mat[recv_l] = recv_staging

    return res_mat


def execute_distributed_fft(obj: FourPoint, comm: MPI.Comm) -> FourPoint:
    """
    Main routine: Call this for objects that are local to a rank but in the respective full BZ slice. E.g., after
    a call of exchange_and_map_irrbz_fullbz.
    This routine performs a distributed 3D FFT by redistributing the data into pencil decompositions for
    each dimension, performing local FFTs, and then redistributing back to the original layout.
    The final result is that obj.mat is transformed in-place to the Fourier space representation
    corresponding to the full BZ.
    Attention: modifies the object in-place!
    """
    nq = obj.nq
    nx, ny, nz = nq

    # --- STEP 1: Z-FFT ---
    # Move to Z-pencils. The number of rows in obj.mat will now be (my_z_pencils * nz)
    obj.mat = _redistribute_p2p(obj.mat, nq, comm, "flat", "z_pencil")

    # Save the shape of the Z-pencil layout to restore after FFT
    shape_z = obj.mat.shape
    # Reshape to (n_pencils, nz, orbitals..., frequencies...)
    obj.mat = obj.mat.reshape(-1, nz, *shape_z[1:])
    fft.fftn(obj.mat, axes=(1,), overwrite_x=True)
    obj.mat = obj.mat.reshape(shape_z)

    # --- STEP 2: Y-FFT ---
    obj.mat = _redistribute_p2p(obj.mat, nq, comm, "z_pencil", "y_pencil")

    shape_y = obj.mat.shape
    obj.mat = obj.mat.reshape(-1, ny, *shape_y[1:])
    fft.fftn(obj.mat, axes=(1,), overwrite_x=True)
    obj.mat = obj.mat.reshape(shape_y)

    # --- STEP 3: X-FFT ---
    obj.mat = _redistribute_p2p(obj.mat, nq, comm, "y_pencil", "x_pencil")

    shape_x = obj.mat.shape
    obj.mat = obj.mat.reshape(-1, nx, *shape_x[1:])
    fft.fftn(obj.mat, axes=(1,), overwrite_x=True)
    obj.mat = obj.mat.reshape(shape_x)

    # --- STEP 4: BACK TO FLAT ---
    obj.mat = _redistribute_p2p(obj.mat, nq, comm, "x_pencil", "flat")

    return obj
