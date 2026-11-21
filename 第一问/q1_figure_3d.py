"""Orthographic companion to the static T1 ground-area figure."""
from functools import lru_cache
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from q1_figure import COLORS, HEIGHTS, PITCH_LIMIT_DEG, PITCH_RADIUS, RECTANGLES, ROOT, TX, TY, raster


WIDTH, HEIGHT = 1540, 1040
INK = "#233843"
ZOOM = 2
az, el = map(math.radians, (32, 36))
u = np.array([math.cos(az), math.sin(az), 0])
v = np.array([math.sin(az)*math.sin(el), -math.cos(az)*math.sin(el), -math.cos(el)])
basis = 9.0*np.vstack([u,v])
origin = np.array([120.,618.])


def point(x,y,z=0):
    return tuple(ZOOM*(origin+basis @ [x,y,z]))


@lru_cache()
def font(size,bold):
    return ImageFont.truetype("C:/Windows/Fonts/msyh"+("bd" if bold else "")+".ttc",size*ZOOM)


def draw():
    # Map each screen pixel back onto the ground plane using the same projection
    # as the vectors below. Outside the square the canvas stays white.
    n=1600
    lookup=np.array([tuple(bytes.fromhex(c[1:])) for c in COLORS],dtype='uint8')
    ground=Image.fromarray(lookup[raster(n)])
    inverse=np.diag([n/100,-n/100]) @ np.linalg.inv(basis[:,:2])
    intercept=np.array([0,n])-inverse @ origin
    transform=np.column_stack([inverse/ZOOM,intercept])
    im=ground.transform((WIDTH*ZOOM,HEIGHT*ZOOM),Image.Transform.AFFINE,
                        tuple(transform.flatten()),Image.Resampling.NEAREST,fillcolor='white')
    d=ImageDraw.Draw(im)

    def text(x,y,s,size=20,color=INK,bold=False,anchor='la'):
        d.text((x*ZOOM,y*ZOOM),s,font=font(size,bold),fill=color,anchor=anchor)

    def line(a,b,color=INK,w=1):
        d.line((a,b),fill=color,width=w*ZOOM)

    def label(p,s,dx=0,dy=0,size=19,color=INK,bold=False,anchor='mm'):
        text(p[0]/ZOOM+dx,p[1]/ZOOM+dy,s,size,color,bold,anchor)

    text(66,34,"T1 的三维可观测区域",44,bold=True)
    text(69,108,"地面颜色表示车心位于该处时的观测条件，楼体按题设高度绘制。",23)
    text(965,42,f"最优水平距离   {PITCH_RADIUS:.4f} m",28,bold=True)
    text(965,92,f"对应俯仰角       {PITCH_LIMIT_DEG}°",26,bold=True)
    corners=[point(0,0),point(100,0),point(100,100),point(0,100)]
    d.line(corners+[corners[0]],fill="#71828a",width=2*ZOOM)
    for i in range(0,101,20):
        p=point(i,0)
        line(p,(p[0]-3*ZOOM,p[1]+7*ZOOM))
        label(p,str(i),-4,19,16)
        p=point(100,i)
        line(p,(p[0]+5*ZOOM,p[1]+5*ZOOM))
        if i>0:
            label(p,str(i),18,10,16)
    label(point(45,0),"向东  x / m",-25,49,19)
    label(point(100,53),"向北  y / m",87,21,19)
    line(point(0,0),point(0,0,46),"#73838c")
    for z in [10,20,30,40]:
        p=point(0,0,z)
        line(p,(p[0]-7*ZOOM,p[1]),"#73838c")
        label(p,str(z),-23,0,16)
    label(point(0,0,46),"高度  z / m",0,-19,18)

    # Draw opaque objects from far to near. The two visible walls face +x and -y.
    objects=[]
    for i,((x0,x1,y0,y1),h) in enumerate(zip(RECTANGLES,HEIGHTS),1):
        x,y=(x0+x1)/2,(y0+y1)/2
        objects.append((x*math.sin(az)-y*math.cos(az),i,(x0,x1,y0,y1,h)))
    objects.append((TX*math.sin(az)-TY*math.cos(az),0,None))
    for _,i,g in sorted(objects):
        if i==0:
            for z in np.arange(0,45,2.15):
                line(point(TX,TY,z),point(TX,TY,min(z+1.15,45)),"#743b4b",2)
            p=point(TX,TY)
            r=4*ZOOM
            d.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill="white",outline="#743b4b",width=2*ZOOM)
            p=point(TX,TY,45)
            r=7*ZOOM
            d.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill="#963951",outline="white",width=2*ZOOM)
            label(p,"T1   45 m",15,-1,24,"#813449",True,"lm")
            continue
        x0,x1,y0,y1,h=g
        for face,color in [
            ([(x0,y0,0),(x1,y0,0),(x1,y0,h),(x0,y0,h)],"#637887"),
            ([(x1,y0,0),(x1,y1,0),(x1,y1,h),(x1,y0,h)],"#435965"),
            ([(x0,y0,h),(x1,y0,h),(x1,y1,h),(x0,y1,h)],"#a6b4bd")]:
            pts=[point(*p) for p in face]
            d.polygon(pts,fill=color)
            d.line(pts+[pts[0]],fill="#495c68",width=ZOOM)
        label(point((x0+x1)/2,(y0+y1)/2,h),f"B{i}",size=19,bold=True)
        label(point((x0+x1)/2,y0,h*0.55),f"{h} m",size=16,color="white")
    y=966
    line((65*ZOOM,(y-17)*ZOOM),((WIDTH-65)*ZOOM,(y-17)*ZOOM),"#d6dde0")
    for x,c,s in zip([66,315,643,948,1280],COLORS,
                     ["可观测","车体净空不足或越界","视线受建筑遮挡","超过 80° 俯仰上限","建筑占地"]):
        d.rectangle((x*ZOOM,(y+6)*ZOOM,(x+27)*ZOOM,(y+29)*ZOOM),fill=c)
        text(x+39,y,s,21)
    im.resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS).save(ROOT/"T1可观测区域_三维.png")


if __name__=="__main__":
    draw()
