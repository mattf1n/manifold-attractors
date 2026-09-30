"""Original intrinsic-area and periodic-region algorithms (see MAPS.md)."""
import numpy as np

def surface_area_map(states,evaluate,grid_size=61,step_degrees=.05,center_kind='middle'):
    """Intrinsic area ratio using orthonormal surface tangents, not chart vectors.

    The original middle state fixes radius. Pole is either that state's direction
    or the normalized raw-vector centroid; first/last give Gram-Schmidt axes.
    Central perturbations remain on sphere; divide by INPUT arc length. Pole
    tangent bases rotate with longitude, but the area metric is basis-invariant.
    """
    assert grid_size>=3 and grid_size%2==1 and step_degrees>0
    values=states.astype(np.float64);mid=len(values)//2
    radius=np.linalg.norm(values[mid])
    assert center_kind in ['middle','centroid']
    defining_indices=[0,mid,len(values)-1]
    center=values[mid] if center_kind=='middle' else values[defining_indices].mean(axis=0)
    assert np.linalg.norm(center)>1e-12
    pole=center/np.linalg.norm(center)
    first=values[0]-(values[0]@pole)*pole
    assert np.linalg.norm(first)>np.linalg.norm(values[0])*1e-8
    first/=np.linalg.norm(first)
    second=values[-1]-(values[-1]@pole)*pole-(values[-1]@first)*first
    residual_norm=float(np.linalg.norm(second))
    assert residual_norm>np.linalg.norm(values[-1])*1e-8,'Degenerate defining directions; do not silently change points'
    second/=np.linalg.norm(second)
    basis=np.stack([pole,first,second]);np.testing.assert_allclose(basis@basis.T,np.eye(3),atol=1e-10)
    longitude=np.linspace(-90,90,grid_size);latitude=np.linspace(-90,90,grid_size)
    phi,psi=np.meshgrid(np.deg2rad(longitude),np.deg2rad(latitude),indexing='xy')
    phi,psi=phi.ravel(),psi.ravel()
    meridian=np.cos(phi)[:,None]*pole+np.sin(phi)[:,None]*first
    normal=np.cos(psi)[:,None]*meridian+np.sin(psi)[:,None]*second
    tangent1=-np.sin(phi)[:,None]*pole+np.cos(phi)[:,None]*first
    tangent2=-np.sin(psi)[:,None]*meridian+np.cos(psi)[:,None]*second
    for vector in [normal,tangent1,tangent2]:np.testing.assert_allclose(np.linalg.norm(vector,axis=1),1,atol=1e-10)
    assert np.max(np.abs((tangent1*tangent2).sum(1)))<1e-10
    def metric(delta_degrees):
        delta=np.deg2rad(delta_degrees);columns=[]
        for tangent in [tangent1,tangent2]:
            plus=radius*(np.cos(delta)*normal+np.sin(delta)*tangent)
            minus=radius*(np.cos(delta)*normal-np.sin(delta)*tangent)
            np.testing.assert_allclose(np.linalg.norm(plus,axis=1),radius,rtol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(minus,axis=1),radius,rtol=1e-10)
            columns.append((evaluate(plus)-evaluate(minus))/(2*radius*delta))
        j1,j2=columns
        g11=(j1*j1).sum(1);g22=(j2*j2).sum(1);g12=(j1*j2).sum(1)
        assert (g11>0).all()
        perpendicular=j2-(g12/g11)[:,None]*j1
        area=np.sqrt(g11)*np.linalg.norm(perpendicular,axis=1)
        gram=np.stack([g11,g12,g12,g22],axis=1).reshape(-1,2,2)
        sigma_max=np.sqrt(np.linalg.eigvalsh(gram)[:,-1]);sigma_min=area/sigma_max
        assert np.isfinite(area).all() and (area>0).all()
        return area,gram,sigma_max,sigma_min
    h=step_degrees;coarse=metric(h)
    for refinement in range(4):
        fine=metric(h/2)
        relative=np.abs(coarse[0]-fine[0])/np.maximum(fine[0],1e-14)
        if np.quantile(relative,.99)<.01 and relative.max()<.05:break
        if refinement==3:raise ValueError('Surface area finite differences did not converge')
        coarse=fine;h/=2
    area,gram,smax,smin=coarse;shape=(grid_size,grid_size)
    pole_variation=max(float(np.ptp(row)/np.mean(row)) for row in [area.reshape(shape)[0],area.reshape(shape)[-1]])
    assert pole_variation<.01
    unit=values/np.linalg.norm(values,axis=1,keepdims=True);coordinates=unit@basis.T
    offplane=np.linalg.norm(unit-coordinates@basis,axis=1)
    projected=coordinates/np.linalg.norm(coordinates,axis=1,keepdims=True)
    marker_longitude=np.rad2deg(np.arctan2(projected[:,1],projected[:,0]))
    marker_latitude=np.rad2deg(np.arctan2(projected[:,2],np.hypot(projected[:,0],projected[:,1])))
    assert np.all(np.abs(marker_longitude)<=90+1e-8) and np.all(np.abs(marker_latitude)<=90+1e-8)
    assert np.max(offplane[[0,mid,-1]])<1e-10
    checks=radius*normal[[0,len(normal)//2,-1]]
    together=evaluate(checks);separate=np.concatenate([evaluate(point[None,:]) for point in checks])
    np.testing.assert_allclose(together,separate,rtol=1e-9,atol=1e-9)
    arrays=dict(longitude=longitude,latitude=latitude,area_ratio=area.reshape(shape),area_ratio_half=fine[0].reshape(shape),
        gram=gram.reshape(*shape,2,2),sigma_max=smax.reshape(shape),sigma_min=smin.reshape(shape),basis=basis,
        sphere_radius=radius,center_kind=center_kind,chart_center_direction=pole,
        marker_longitude=marker_longitude,marker_latitude=marker_latitude,offplane=offplane)
    metadata=dict(radius=float(radius),middle_index=mid,center_kind=center_kind,
        center_vector_norm=float(np.linalg.norm(center)),defining_indices=defining_indices,
        radius_rule='unchanged norm of middle-index embedding',
        second_tangent_pre_normalization_norm=residual_norm,
        grid_size=grid_size,step_degrees=h,convergence_p99=float(np.quantile(relative,.99)),
        convergence_max=float(relative.max()),pole_relative_variation=pole_variation,
        batch_max_abs=float(np.max(np.abs(together-separate))),max_offplane_degrees=float(np.rad2deg(np.arcsin(offplane.max()))),
        area_min=float(area.min()),area_max=float(area.max()),area_at_chart_center=float(area.reshape(shape)[grid_size//2,grid_size//2]),
        definition='sqrt(det(J_t.T J_t)) for orthonormal INPUT surface tangents; output area / input area',
        chart='r*(cos(latitude)*(cos(longitude)*pole+sin(longitude)*e1)+sin(latitude)*e2); both angles -90..90',
        frame='pole is normalized chosen center; e1 toward first, e2 third component orthogonal to pole/e1; no PCA')
    return arrays,metadata
def extend_centroid_surface_map(original, evaluate, step_degrees=.05, recompute_all=False):
    """Extend a frozen ±90° surface, or recompute its full sphere for a new F.

    By default only missing columns are evaluated. ``recompute_all=True`` uses
    the frozen frame but evaluates every cell, never reusing the old areas.
    This exploration-only extension leaves the paper helper and its pinned hashes
    untouched. It uses that helper's intrinsic great-circle FD metric verbatim,
    but accepts the frozen frame and a rectangular longitude domain.
    """
    basis=original['basis'];radius=float(original['sphere_radius'])
    np.testing.assert_allclose(basis@basis.T,np.eye(3),rtol=0,atol=1e-12)
    longitude=np.linspace(-180,180,289);latitude=original['latitude'].copy()
    np.testing.assert_array_equal(longitude[72:217],original['longitude'])
    np.testing.assert_array_equal(latitude,np.linspace(-90,90,145))
    missing=np.arange(289) if recompute_all else np.r_[0:72,217:289]
    phi,psi=np.meshgrid(np.deg2rad(longitude[missing]),np.deg2rad(latitude),indexing='xy')
    phi,psi=phi.ravel(),psi.ravel()
    pole,first,second=basis
    meridian=np.cos(phi)[:,None]*pole+np.sin(phi)[:,None]*first
    normal=np.cos(psi)[:,None]*meridian+np.sin(psi)[:,None]*second
    tangent1=-np.sin(phi)[:,None]*pole+np.cos(phi)[:,None]*first
    tangent2=-np.sin(psi)[:,None]*meridian+np.cos(psi)[:,None]*second
    for vector in [normal,tangent1,tangent2]:np.testing.assert_allclose(np.linalg.norm(vector,axis=1),1,atol=1e-10)
    assert np.max(np.abs((tangent1*tangent2).sum(1)))<1e-10
    def metric(delta_degrees):
        delta=np.deg2rad(delta_degrees);columns=[]
        for tangent in [tangent1,tangent2]:
            plus=radius*(np.cos(delta)*normal+np.sin(delta)*tangent)
            minus=radius*(np.cos(delta)*normal-np.sin(delta)*tangent)
            np.testing.assert_allclose(np.linalg.norm(plus,axis=1),radius,rtol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(minus,axis=1),radius,rtol=1e-10)
            columns.append((evaluate(plus)-evaluate(minus))/(2*radius*delta))
        j1,j2=columns
        g11=(j1*j1).sum(1);g22=(j2*j2).sum(1);g12=(j1*j2).sum(1)
        assert (g11>0).all()
        perpendicular=j2-(g12/g11)[:,None]*j1
        area=np.sqrt(g11)*np.linalg.norm(perpendicular,axis=1)
        gram=np.stack([g11,g12,g12,g22],axis=1).reshape(-1,2,2)
        sigma_max=np.sqrt(np.linalg.eigvalsh(gram)[:,-1]);sigma_min=area/sigma_max
        assert np.isfinite(area).all() and (area>0).all()
        return area,gram,sigma_max,sigma_min
    coarse=metric(step_degrees);fine=metric(step_degrees/2)
    relative=np.abs(coarse[0]-fine[0])/np.maximum(fine[0],1e-14)
    assert np.quantile(relative,.99)<.01 and relative.max()<.05,'Surface FD not converged; do not silently replace old cells'
    arrays={k:np.array(v,copy=True) for k,v in original.items()}
    arrays['longitude']=longitude
    for key,values in zip(['area_ratio','gram','sigma_max','sigma_min','area_ratio_half'],[*coarse,fine[0]]):
        tail=original[key].shape[2:]
        full=np.empty((145,289)+tail,dtype=original[key].dtype)
        if not recompute_all:full[:,72:217]=original[key]
        full[:,missing]=values.reshape((145,len(missing))+tail)
        if not recompute_all:np.testing.assert_array_equal(full[:,72:217],original[key])
        arrays[key]=full
    area=arrays['area_ratio']
    poles=max(float(np.ptp(row)/np.mean(row)) for row in [area[0],area[-1]])
    seam=float(np.max(np.abs(area[:,0]-area[:,-1])/np.maximum(area[:,0],1e-14)))
    assert poles<.01 and seam<1e-7
    test=radius*normal[[0,len(normal)//2,-1]]
    together=evaluate(test);separate=np.concatenate([evaluate(point[None,:]) for point in test])
    np.testing.assert_allclose(together,separate,rtol=1e-9,atol=1e-9)
    checks=dict(grid_shape=[145,289],new_columns=len(missing),new_points=145*len(missing),old_grid_columns=[72,216],old_hemisphere_bit_exact=not recompute_all,step_degrees=step_degrees,convergence_p99=float(np.quantile(relative,.99)),convergence_max=float(relative.max()),pole_relative_variation=poles,seam_relative_max=seam,batch_max_abs=float(np.max(np.abs(together-separate))),input_radius_checks=True,orthonormal_tangents=True,metric='Intrinsic output/input orthonormal tangent area; no display-projection Jacobian')
    return arrays,checks
def tangent(direction, x):
    x = x.astype(np.float64)
    chat = x / np.linalg.norm(x)
    u = direction - (direction @ chat) * chat
    retained = float(np.linalg.norm(u))
    assert retained > 1e-8
    u /= retained
    checks = dict(third_tangent_orthogonality=float(abs(chat @ u)),
                  tangent_unit_error=float(abs(np.linalg.norm(u)-1)),
                  theta_zero_max_abs=float(np.max(np.abs(np.linalg.norm(x)*chat-x))),
                  projected_direction_norm=retained)
    assert max(checks[k] for k in checks if k != 'projected_direction_norm') < 1e-10
    return u, checks

def regions(ids, step):
    """Periodic Voronoi-cell intervals; each sampled token covers exactly one cell."""
    ids = np.asarray(ids)
    n = len(ids)
    assert np.isclose(n*step, 360)
    changes = np.flatnonzero(ids != np.roll(ids, 1))
    if not len(changes):
        return [dict(start=-step/2, end=360-step/2, token_id=int(ids[0]), samples=n)]
    out = []
    for j, first in enumerate(changes):
        last = int(changes[(j+1) % len(changes)])
        if last <= first:
            last += n
        out.append(dict(start=float(first*step-step/2), end=float(last*step-step/2),
                        token_id=int(ids[first]), samples=int(last-first)))
    assert sum(x['samples'] for x in out) == n
    assert abs(sum(x['end']-x['start'] for x in out)-360) < 1e-8
    # All samples assigned exactly once and identity preserved, including seam.
    reconstructed = np.empty(n, dtype=ids.dtype)
    hits = np.zeros(n, int)
    for r in out:
        first = int(round((r['start']+step/2)/step))
        ix = np.arange(first, first+r['samples']) % n
        reconstructed[ix] = r['token_id']; hits[ix] += 1
    np.testing.assert_array_equal(hits, 1)
    np.testing.assert_array_equal(reconstructed, ids)
    return out

def frame_from_point(point, direction, seed):
    radius = np.linalg.norm(point); pole = point / radius
    e1 = direction - (direction @ pole) * pole
    if np.linalg.norm(e1) < 1e-10: raise ValueError('Steering direction is degenerate at this state')
    e1 /= np.linalg.norm(e1)
    random = np.random.default_rng(seed).normal(size=len(point)); random /= np.linalg.norm(random)
    e2 = random - (random @ pole) * pole - (random @ e1) * e1
    if np.linalg.norm(e2) < 1e-10: raise ValueError('Orthogonal axis is degenerate; do not silently change seed')
    e2 /= np.linalg.norm(e2); basis = np.stack([pole, e1, e2])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    return dict(point=point, basis=basis, sphere_radius=np.array(radius))
