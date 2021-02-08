
import xarray as xr
from torch.utils.data import Dataset, DataLoader

from data.transformation import Normalize

# Craete dummy xarray data.

ds = xr.Dataset()
for x in range(3):
    # Random data with different mean (=x)
    ds[f'x{x}'] = xr.DataArray(np.random.normal(loc=x, scale=10., size=(seq_len, num_sites)).astype('float32'), dims=['time', 'site'], coords={'time': range(seq_len), 'site': [f'S{s}' for s in range(3)]})

print(ds)

# Split into training and test set.

train_ds = ds.sel(site='S0').isel(time=range(250))
valid_ds = ds.sel(site='S0').isel(time=range(250, 500))

# Create a normalizer using training data.

norm = Normalize()
norm.register_dict({
    'x0': train_ds.x0.values,
    'x1': train_ds.x1.values,
    'x2': train_ds.x2.values,
})
print(norm)

# Build a simple pytorch dataset.

class SiteData(Dataset):
    def __init__(self, ds, dtype=np.float32):
        super().__init__()
        
        self.ds = ds
        self.dtype = dtype

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, ind):
        d = self.ds.isel(time=ind)
        return {
            'x0': d['x0'].values.astype(self.dtype),
            'x1': d['x1'].values.astype(self.dtype),
            'x2': d['x2'].values.astype(self.dtype),
        }

data = SiteData(train_ds, n)

# Load one sample.

d = data[0]
print(d)

# Normalize sample.

d_norm = norm.normalize_dict(d, return_stack=True)
print(d_norm)

# Buid a pytorch dataloader.

dl = DataLoader(data, batch_size=8)

# Get first batch.

batch = next(iter(dl))

# Access data;

x0 = batch['x0']
x1 = batch['x1']
x2 = batch['x2']

# Validity check.

print(torch.isclose(x2, norm.unnormalize('x2', norm.normalize('x2', x2))))

# Print mean and std.

def print_ms(x):
    if not isinstance(x, dict):
        x = {'x': x}
    for key, val in x.items():
        print(f'{key}: mean={val.mean()}, std={val.std()}')

# Normalize single variable.

print_ms(norm.normalize('x2', x2))

# Normalize entire batch.

print_ms(norm.normalize_dict(batch))

# Normalize and stack along last dimension.


















