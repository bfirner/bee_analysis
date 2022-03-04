# Should be run with pytest:
# > python3 -m pytest
# Or, if you want to print things out and see then even when a test is passing:
# > python3 -m pytest -rP

# Adding a test to your code makes it very professional and also makes it possible that your code
# may even work.


import torch
import shapenet
from datagen.CreateShapeData import all_shapes

def test_chunking():
    """A test for shape chunking."""

    all_chunks = shapenet.chunkifyShapes(shapes=all_shapes.values(), stride=3)

    # All of the chunks, when sampled with stride 3, should reassemble into the original shapes.
    stride = 3
    for idx in range(len(all_shapes)):
        coords, chunks = all_chunks[idx]
        #print(f"chunks from {list(all_shapes.keys())[idx]} are {chunks}")
        original = list(all_shapes.values())[idx]
        reproduction = torch.zeros(original.size()).expand(1, -1, -1)
        for coord, chunk in zip(coords, chunks):
            x, y = coord
            x = x * stride
            y = y * stride
            # Only take part of the chunk if we run out of pixels in the original shape.
            reproduction[:, y:y+stride, x:x+stride] = chunk[:, :original.size(0) - y, :original.size(1) - x]
        assert (original == reproduction).all()

def test_nn_solver():
    """A test for shapenet solution."""
    used_shapes = list(all_shapes.values())
    net = shapenet.ShapeNet(in_dimensions = (1, 20, 20), target_shapes=used_shapes)

    test_input = torch.zeros(1, 1, 20, 20)
    for idx, shape in enumerate(used_shapes):
        #print(f"Testing with shape {shape}")
        test_input.fill_(0.)
        test_input[0, 0, :shape.size(0), :shape.size(1)] = shape
        out = net.forward(test_input)[0]
        #print(f"input: {test_input}")
        #rint(f"output: {out}")
        # Only one thing should have been detected and it should be this shape.
        assert out.sum() == 1.
        assert out[idx] == 1.


