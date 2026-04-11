from __future__ import annotations

import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Find chunk boundaries that do not split a designated special token."""
    # The split token must be bytes because the file is read in binary mode.
    if not isinstance(split_special_token, bytes):
        raise TypeError("split_special_token must be a bytestring")
    if desired_num_chunks <= 0:
        raise ValueError("desired_num_chunks must be positive")

    # Seek to the end to measure the file size.
    file.seek(0, os.SEEK_END)
    file_size = file.tell()

    # Reset to the start so future callers see a consistent file position.
    file.seek(0)

    # Use roughly uniform initial boundary guesses.
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [index * chunk_size for index in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    # Scan forward in small windows until each boundary aligns to the special token.
    mini_chunk_size = 4096

    for boundary_index in range(1, len(chunk_boundaries) - 1):
        search_position = chunk_boundaries[boundary_index]
        file.seek(search_position)

        while True:
            # Read a small binary window starting from the current search position.
            mini_chunk = file.read(mini_chunk_size)

            # If we hit EOF before finding the split token, clamp to the file end.
            if mini_chunk == b"":
                chunk_boundaries[boundary_index] = file_size
                break

            # Look for the next occurrence of the special token in this window.
            offset = mini_chunk.find(split_special_token)
            if offset != -1:
                chunk_boundaries[boundary_index] = search_position + offset
                break

            # Otherwise keep scanning forward.
            search_position += mini_chunk_size

    # Boundaries can collapse together near EOF, so return the sorted unique set.
    return sorted(set(chunk_boundaries))
