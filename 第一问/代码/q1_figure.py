"""Draw the contest's static T1 cross-section and check its area independently."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "综合" / "代码"))
from check_bounds import RECTANGLES, RADIUS


TX, TY = 50 + 20 * math.sqrt(3), 70
HEIGHTS = [20, 28, 12, 24, 32]
PITCH_LIMIT_DEG = 80
PITCH_RADIUS = 43 / math.tan(math.radians(PITCH_LIMIT_DEG))
COLORS = ["#85cbb1", "#d2d9df", "#e1ac73", "#b09bd2", "#485663"]
IMAGE_DIR = Path(__file__).resolve().parents[1] / "图片"


def classify(x, y):
    """Categories: observable, vehicle exclusion, blocked, pitch, footprint."""
    allowed = (x >= RADIUS) & (x <= 100-RADIUS) & (y >= RADIUS) & (y <= 100-RADIUS)
    blocked = np.zeros_like(allowed)
    footprint = np.zeros_like(allowed)
    for (x0,x1,y0,y1), roof in zip(RECTANGLES, HEIGHTS):
        dx = np.maximum(np.maximum(x0-x,x-x1),0)
        dy = np.maximum(np.maximum(y0-y,y-y1),0)
        allowed &= dx**2+dy**2 >= RADIUS**2
        footprint |= (dx==0) & (dy==0)
        # Segment from camera to T1. Only the part at or below roof height can hit.
        lo, hi = np.zeros_like(allowed,dtype=float), np.full(allowed.shape,(roof-2)/43)
        for start, stop, left, right in ((x,TX,x0,x1),(y,TY,y0,y1)):
            direction = stop-start
            parallel = direction==0
            safe = np.where(parallel,1,direction)
            a,b = (left-start)/safe,(right-start)/safe
            lo = np.maximum(lo,np.where(parallel,-np.inf,np.minimum(a,b)))
            hi = np.minimum(hi,np.where(parallel,np.inf,np.maximum(a,b)))
            hi = np.where(parallel & ((start<left)|(start>right)),-np.inf,hi)
        blocked |= lo <= hi
    steep = (TX-x)**2+(TY-y)**2 < PITCH_RADIUS**2
    return np.select([footprint,~allowed,blocked,steep],[4,1,2,3],default=0).astype('uint8')


def hull(points):
    def half(ordered):
        chain = []
        for p in ordered:
            while len(chain)>=2:
                a,b = chain[-2:]
                if (b[0]-a[0])*(p[1]-a[1]) > (b[1]-a[1])*(p[0]-a[0]):
                    break
                chain.pop()
            chain.append(p)
        return chain
    ordered = sorted(set(points))
    return half(ordered)[:-1]+half(reversed(ordered))[:-1]


def section_area(step):
    """Independent vertical integration of rounded rectangles, shadows and circle."""
    xs = np.arange(1.5+step/2,98.5,step)
    intervals = []
    def add(low,high):
        low, high = np.clip(low,1.5,98.5), np.clip(high,1.5,98.5)
        intervals.append((low,np.maximum(low,high)))
    for (x0,x1,y0,y1),roof in zip(RECTANGLES, HEIGHTS):
        dx = np.maximum(np.maximum(x0-xs,xs-x1),0)
        cap = np.sqrt(np.maximum(RADIUS**2-dx**2,0))
        add(np.where(dx<=RADIUS,y0-cap,0),np.where(dx<=RADIUS,y1+cap,0))
        corners=[(x,y) for x in (x0,x1) for y in (y0,y1)]
        k=43/(45-roof)
        polygon=hull(corners+[(TX+k*(x-TX),TY+k*(y-TY)) for x,y in corners])
        low,high=np.full_like(xs,np.inf),np.full_like(xs,-np.inf)
        for a,b in zip(polygon,polygon[1:]+polygon[:1]):
            if a[0]==b[0]:
                continue
            active=(xs>=min(a[0],b[0])) & (xs<=max(a[0],b[0]))
            ordinate=a[1]+(xs-a[0])*(b[1]-a[1])/(b[0]-a[0])
            low=np.minimum(low,np.where(active,ordinate,np.inf))
            high=np.maximum(high,np.where(active,ordinate,-np.inf))
        add(low,high)
    cap=np.sqrt(np.maximum(PITCH_RADIUS**2-(xs-TX)**2,0))
    add(TY-cap,TY+cap)
    lows,highs=np.array(intervals).transpose(1,0,2)
    order=np.argsort(lows,axis=0)
    lows,highs=[np.take_along_axis(v,order,axis=0) for v in (lows,highs)]
    end=np.full_like(xs,1.5)
    covered=np.zeros_like(xs)
    for low,high in zip(lows,highs):
        covered+=np.maximum(0,high-np.maximum(end,low))
        end=np.maximum(end,high)
    return float((97-covered).sum()*step)


def raster(n):
    centers=(np.arange(n)+0.5)*100/n
    return np.vstack([classify(centers[None,:],row[:,None]) for row in np.array_split(centers[::-1],20)])


def draw(data,area):
    canvas=Image.new("RGB",(1320,1000),"white")
    d=ImageDraw.Draw(canvas)
    def label(x,y,s,size=23,bold=False,anchor=None):
        font=ImageFont.truetype("C:/Windows/Fonts/msyh"+("bd" if bold else "")+".ttc",size)
        d.text((x,y),s,font=font,fill="#253741",anchor=anchor)
    label(80,30,"T1 的可观测区域",42,True)
    label(82,96,"第一问   /   车心位置的水平投影",23)
    left,top,size=93,163,720
    def pos(x,y):
        return left+size*x/100,top+size*(1-y/100)
    rgb=np.array([tuple(bytes.fromhex(c[1:])) for c in COLORS],dtype=np.uint8)
    canvas.paste(Image.fromarray(rgb[data]).resize((size,size),Image.Resampling.NEAREST),(left,top))
    d.rectangle([pos(0,100),pos(100,0)],outline="#516371",width=2)
    for t in range(0,101,20):
        px,py=pos(t,t)
        d.line((px,top+size,px,top+size+7),fill="#516371",width=2)
        label(px,top+size+12,str(t),20,anchor="mt")
        d.line((left-7,py,left,py),fill="#516371",width=2)
        label(left-13,py,str(t),20,anchor="rm")
    label(left+size/2,944,"向东  x（m）",22,anchor="mt")
    label(left,top-29,"向北  y（m）",18)
    for i,(x0,x1,y0,y1) in enumerate(RECTANGLES,1):
        x,y=pos((x0+x1)/2,(y0+y1)/2)
        font=ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc",20)
        d.text((x,y),"B"+str(i),font=font,fill="white",anchor="mm")
    x,y=pos(TX,TY)
    d.ellipse((x-5,y-5,x+5,y+5),fill="#203944",outline="white",width=1)
    label(x+9,y-18,"T1",18,True)
    label(868,182,"可观测面积",23)
    label(864,224,f"{area:,.1f} m²",45,True)
    label(868,303,f"占整个场地的 {area/100:.2f}%",22)
    for i,s in enumerate(("可观测","车体净空不足或越界","视线受建筑遮挡","超过 80° 俯仰上限","建筑占地")):
        y=412+i*57
        d.rectangle((868,y+3,901,y+30),fill=COLORS[i])
        label(917,y,s,23)
    d.line((868,712,1240,712),fill="#d6dde0",width=2)
    label(868,740,f"最优水平距离   {PITCH_RADIUS:.4f} m",25,True)
    label(868,794,f"对应俯仰角       {PITCH_LIMIT_DEG}°",25,True)
    label(868,860,"相机高 2 m，目标高 45 m",21)
    path = IMAGE_DIR / "T1可观测区域.png"
    canvas.save(path)
    return path


if __name__=="__main__":
    data=raster(2000)
    areas={str(step):section_area(step) for step in (0.005,0.001,0.0005,0.00025)}
    area=areas["0.00025"]
    grid=float(np.count_nonzero(data==0)*0.05**2)
    assert abs(areas["0.0005"]-area)<0.001 and abs(grid-area)<1
    image_path = draw(data,area)
    result=dict(image_path=str(image_path),section_areas_m2=areas,grid_005m_area_m2=grid,pitch_radius_m=PITCH_RADIUS,
                share_of_site=area/10000,share_of_vehicle_free_area=area/8302.657082647)
    print(json.dumps(result,ensure_ascii=False,indent=2))
