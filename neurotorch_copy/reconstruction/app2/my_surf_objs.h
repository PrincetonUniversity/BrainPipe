// OCT 11, modification :
// 1.add degree as union type in MyMarker,
// 2. MyMarker is alias as MyNode
// 3. just disable __SET_MARKER_DEGREE__ if you don't like these change

//last change; by PHC 20121127. update the swc and marker saving funcs

#ifndef __MY_SURF_OBJS_H__
#define __MY_SURF_OBJS_H__
#include <vector>
#include <list>
#include <map>
#include <iostream>
#include <fstream>
#include <cassert>
#include "color_xyz.h"
#define V3DLONG long
#define __SET_MARKER_DEGREE__
#define __ESWC__

using namespace std;

#define MARKER_BASE 1.0 // the basic offset of marker is 1.0, the marker coordinate will be converted when read and save


struct BasicSurfObj {
    V3DLONG n;				// index
    RGBA8 color;
    bool on;
    bool selected;
    // QString name;
    // QString comment;
    BasicSurfObj() {
        n=0;
        color.r=color.g=color.b=color.a=255;
        on=true;
        selected=false;
        // name=comment="";
    }
};

struct NeuronSWC : public BasicSurfObj {
    int type;			// 0-Undefined, 1-Soma, 2-Axon, 3-Dendrite, 4-Apical_dendrite, 5-Fork_point, 6-End_point, 7-Custom
    float x, y, z;		// point coordinates

    union {
        float r;			// radius
        float radius;
    };

    union {
        V3DLONG pn;				// previous point index (-1 for the first point)
        V3DLONG parent;				// previous point index (-1 for the first point)
    };

    V3DLONG level; //20120217, by PHC. for ESWC format
  //QList<float> fea_val; //20120217, by PHC. for ESWC format

    V3DLONG seg_id; //this is reused for ESWC format, 20120217, by PHC
    V3DLONG nodeinseg_id; //090925, 091027: for segment editing

    operator XYZ() const {
        return XYZ(x, y, z);
    }
    NeuronSWC () {
        n=type=pn=0;
        x=y=z=r=0;
        seg_id=-1;
        nodeinseg_id=0;
        //fea_val=QList<float>();
        level=-1;
    }
};

struct MyPoint {
    int x;
    int y;
    int z;
    bool operator<(const MyPoint & other) const {
        if(z > other.z) return false;
        if(z < other.z) return true;
        if(y > other.y) return false;
        if(y < other.y) return true;
        if(x > other.x) return false;
        if(x < other.x) return true;
        return false;
    }
    MyPoint() {
        x = 0;
        y = 0;
        z = 0;
    }
    MyPoint(int _x, int _y, int _z) {
        x = _x;
        y = _y;
        z = _z;
    }
};

struct MYXYZ {
    double x,y,z;
};

// root node with parent 0
struct MyMarker {
    double x;
    double y;
    double z;
    union {
#ifdef __SET_MARKER_DEGREE__
        double degree;
#endif
        double radius;
    };
    int type;
    MyMarker* parent;
    MyMarker() {
        x=y=z=radius=0.0;
        type = 3;
        parent=0;
    }
    MyMarker(double _x, double _y, double _z) {
        x = _x;
        y = _y;
        z = _z;
        radius = 0.0;
        type = 3;
        parent = 0;
    }
    MyMarker(const MyMarker & v) {
        x=v.x;
        y=v.y;
        z=v.z;
        radius = v.radius;
        type = v.type;
        parent = v.parent;
    }
    MyMarker(const MyPoint & v) {
        x=v.x;
        y=v.y;
        z=v.z;
        radius = 0.0;
        type = 3;
        parent = 0;
    }
    MyMarker(const MYXYZ & v) {
        x=v.x;
        y=v.y;
        z=v.z;
        radius = 0.0;
        type = 3;
        parent = 0;
    }

    double & operator [] (const int i) {
        assert(i>=0 && i <= 2);
        return (i==0) ? x : ((i==1) ? y : z);
    }

    bool operator<(const MyMarker & other) const {
        if(z > other.z) return false;
        if(z < other.z) return true;
        if(y > other.y) return false;
        if(y < other.y) return true;
        if(x > other.x) return false;
        if(x < other.x) return true;
        return false;
    }
    bool operator==(const MyMarker & other) const {
        return (z==other.z && y==other.y && x==other.x);
    }
    bool operator!=(const MyMarker & other) const {
        return (z!=other.z || y!=other.y || x!=other.x);
    }

    long long ind(long long sz0, long long sz01) {
        return ((long long)(z+0.5) * sz01 + (long long)(y+0.5)*sz0 + (long long)(
                    x+0.5));
    }
};

struct MyMarkerX : public MyMarker {
    double feature;
    int seg_id;
    int seg_level;
    MyMarkerX() : MyMarker() {
        seg_id = -1;
        seg_level = -1;
        feature = 0.0;
    }
    MyMarkerX(MyMarker & _marker) {
        x = _marker.x;
        y = _marker.y;
        z = _marker.z;
        type = _marker.type;
        radius = _marker.radius;
        seg_id = -1;
        seg_level = -1;
        feature = 0.0;
    }
    MyMarkerX(double _x, double _y, double _z) : MyMarker(_x, _y, _z) {
        seg_id = -1;
        seg_level = -1;
        feature = 0.0;
    }
};
typedef MyMarker MyNode;

#define MidMarker(m1, m2) MyMarker(((m1).x + (m2).x)/2.0,((m1).y + (m2).y)/2.0,((m1).z + (m2).z)/2.0)


vector<MyMarker> readMarker_file(string marker_file);
bool readMarker_file(string marker_file, vector<MyMarker*> &markers);
bool saveMarker_file(string marker_file, vector<MyMarker> & out_markers);
bool saveMarker_file(string marker_file, vector<MyMarker> & outmarkers,
                     list<string> & infostring);
bool saveMarker_file(string marker_file, vector<MyMarker*> & out_markers);
bool saveMarker_file(string marker_file, vector<MyMarker*> & out_markers,
                     list<string> & infostring);

vector<MyMarker*> readSWC_file(string swc_file);
bool readSWC_file(string swc_file, vector<MyMarker> & outmarkers);
bool saveSWC_file(string swc_file, vector<MyMarker*> & out_markers);
bool saveSWC_file(string marker_file, vector<MyMarker*> & outmarkers,
                  list<string> & infostring);

class NeuronSWC; //from vaa3d's basic_surf_objs.h // added by PHC, 2013-01-03
bool saveSWC_file(string marker_file, vector<NeuronSWC*> & outmarkers,
                  list<string> & infostring);

bool saveDot_file(string swc_file,
                  vector<MyMarker*> & out_markers);              // save graphviz format

#ifdef __ESWC__
bool readESWC_file(string swc_file, vector<MyMarkerX*> &);  // read swc to eswc
bool saveESWC_file(string swc_file, vector<MyMarkerX*> & out_markers);
bool saveESWC_file(string swc_file, vector<MyMarkerX*> & out_markers,
                   list<string> & infostring);
#endif

double dist(MyMarker a, MyMarker b);

vector<MyMarker*> getLeaf_markers(vector<MyMarker*> & inmarkers);
vector<MyMarker*> getLeaf_markers(vector<MyMarker*> & inmarkers,
                                  map<MyMarker *, int> & childs_num);

#endif
