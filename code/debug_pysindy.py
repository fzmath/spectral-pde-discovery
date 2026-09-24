import numpy as np, pysindy as ps, warnings
warnings.filterwarnings('ignore')
data=np.load(r'H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde\burgers\burgers.npz')
u=data['u'][:,:,0]; t=data['t']; x=data['x']
print('u shape:', u.shape, 'x:', len(x), 't:', len(t))
s=x.reshape(-1,1)
lib=ps.PDELibrary(library_functions=[lambda x:x, lambda x:x**2], function_names=[lambda x:'u', lambda x:'u^2'], derivative_order=4, spatial_grid=s, include_bias=False)
opt=ps.SR3(threshold=0.01, max_iter=200)
model=ps.SINDy(feature_library=lib, optimizer=opt)
try:
    model.fit(u, t=t)
    print('Feature names:', model.get_feature_names())
    print('Coefficients shape:', model.coefficients().shape)
    print('Coefficients:', model.coefficients())
    print('Equations:', model.equations())
except Exception as e:
    print('ERROR:', e)
    import traceback; traceback.print_exc()
