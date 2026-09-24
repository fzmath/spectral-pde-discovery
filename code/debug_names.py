import numpy as np, pysindy as ps, warnings
warnings.filterwarnings('ignore')
data=np.load(r'H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde\burgers\burgers.npz')
u=data['u'][:,:,0]; t=data['t']; x=data['x']
print('u shape:', u.shape)  # (256, 101)
u_3d = u[:,:,np.newaxis]  # (256, 101, 1)
s=x.reshape(-1,1)  # (256, 1)
lib=ps.PDELibrary(library_functions=[lambda x:x, lambda x:x**2], function_names=[lambda x:'u', lambda x:'u^2'], derivative_order=4, spatial_grid=s, include_bias=False)
opt=ps.SR3(threshold=0.01, max_iter=200)
model=ps.SINDy(feature_library=lib, optimizer=opt)
model.fit(u_3d, t=t)
print('Feature names:')
for i, f in enumerate(model.get_feature_names()):
    print(f'  [{i}] "{f}" = {model.coefficients()[0][i]:.6f}')
print('Equations:', model.equations())
