import torch

from gns_pytorch import compute_gns


def test_compute_gns_basic():
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1)
    )

    # Create random input and target
    batch_size = 4
    x = torch.randn(batch_size, 10)
    y = torch.randn(batch_size, 1)

    # Compute per-example losses
    outputs = model(x)
    loss_fn = torch.nn.MSELoss(reduction="none")
    loss_per_example = loss_fn(outputs, y).squeeze()

    # Compute GNS without vmap
    gns_value = compute_gns(
        loss_per_example=loss_per_example, model=model, use_vmap=False
    )

    # Check that GNS value is a float and not NaN
    assert isinstance(gns_value, float)
    assert not torch.isnan(torch.tensor(gns_value))
