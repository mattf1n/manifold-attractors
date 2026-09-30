"""Frozen downhill-basin geometry for the paper barrier-strength result.

No uphill moves, smoothing, locality cutoffs, or amplitude thresholds.
Input is the periodic raw output-speed profile in units per radian.
"""
import numpy as np

def periodic_extrema(speed):
    previous,following=np.roll(speed,1),np.roll(speed,-1)
    # One strict side includes a flat extremum boundary deterministically.
    minima=np.flatnonzero((speed<previous)&(speed<=following))
    maxima=np.flatnonzero((speed>previous)&(speed>=following))
    return minima,maxima

def downhill_minimum(speed,grid_step,start_angle):
    """Return reached index/unwrapped angle and a verifiably nonincreasing path."""
    n=len(speed);coordinate=start_angle/grid_step
    nearest=int(round(coordinate))
    if np.isclose(coordinate,nearest,rtol=0,atol=1e-10):
        index=nearest%n;angle=float(nearest*grid_step)
        path_angles=[float(start_angle)];path_values=[float(speed[index])]
    else:
        left=int(np.floor(coordinate));right=left+1;fraction=coordinate-left
        start_value=float((1-fraction)*speed[left%n]+fraction*speed[right%n])
        chosen=left if speed[left%n]<=speed[right%n] else right
        index=chosen%n;angle=float(chosen*grid_step)
        path_angles=[float(start_angle),angle];path_values=[start_value,float(speed[index])]
    for _ in range(n):
        candidates=[((index-1)%n,-1),((index+1)%n,1)]
        lower=[item for item in candidates if speed[item[0]]<speed[index]]
        if not lower:break
        next_index,direction=min(lower,key=lambda item:speed[item[0]])
        index=next_index;angle+=direction*grid_step
        path_angles.append(float(angle));path_values.append(float(speed[index]))
    else:raise AssertionError('Strict descent cannot cycle around a periodic grid')
    assert np.all(np.diff(path_values)<=1e-12*max(path_values))
    assert speed[index]<=speed[(index-1)%n] and speed[index]<=speed[(index+1)%n]
    return index,angle,dict(angles=path_angles,speeds=path_values,
                            start_speed=path_values[0],minimum_speed=float(speed[index]),
                            nonincreasing=True,grid_steps=len(path_values)-1)

def analyze(speed,grid_step,partner_angle):
    angle=np.arange(len(speed))*grid_step
    minima,maxima=periodic_extrema(speed)
    assert len(minima)>0 and len(maxima)>0
    left,aleft,left_path=downhill_minimum(speed,grid_step,0.)
    right,aright,right_path=downhill_minimum(speed,grid_step,partner_angle)
    dleft,dright=aleft,aright-partner_angle
    distinct=left!=right
    ordered=distinct and 0<aright-aleft<360
    peak=None;interior_minima=[];interior_maxima=[]
    if ordered:
        unwrapped=aleft+(angle-aleft)%360
        interior_minima=[int(i) for i in minima if i not in (left,right) and aleft<unwrapped[i]<aright]
        interior_maxima=[int(i) for i in maxima if aleft<unwrapped[i]<aright]
        if interior_maxima:peak=max(interior_maxima,key=lambda i:speed[i])
    topology=bool(ordered and peak is not None and speed[peak]>max(speed[left],speed[right]))
    return dict(anchor_minimum_angle=aleft,partner_minimum_angle=aright,
        anchor_descent=left_path,partner_descent=right_path,
        anchor_minimum_offset_degrees=dleft,partner_minimum_offset_degrees=dright,
        max_absolute_offset_degrees=max(abs(dleft),abs(dright)),
        max_offset_fraction_of_pair_angle=max(abs(dleft),abs(dright))/partner_angle,
        anchor_minimum_speed=float(speed[left]),partner_minimum_speed=float(speed[right]),
        minima_distinct=distinct,minima_ordered=ordered,
        minimum_peak_minimum=topology,
        strict_single_peak_w=bool(topology and not interior_minima and len(interior_maxima)==1),
        interior_minima=len(interior_minima),interior_maxima=len(interior_maxima),
        intervening_peak_angle=float(aleft+(angle[peak]-aleft)%360) if peak is not None else None,
        intervening_peak_speed=float(speed[peak]) if peak is not None else None,
        peak_over_higher_minimum=float(speed[peak]/max(speed[left],speed[right])) if peak is not None else None,
        total_periodic_minima=len(minima),total_periodic_maxima=len(maxima))

def speed_from_vectors(vectors,step,stride):
    derivative=(np.roll(vectors,-stride,axis=0)-np.roll(vectors,stride,axis=0))/(2*np.deg2rad(step*stride))
    return np.linalg.norm(derivative,axis=-1)
