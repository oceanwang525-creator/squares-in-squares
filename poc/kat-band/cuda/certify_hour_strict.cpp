// Independent CPU-only exhaustive SAT screen. No broadphase, epsilon, or repair.
// Long double is a strict numerical screen, NOT an interval proof. Official
// literal and actual pandas3.1.0 metric remain mandatory independent authorities.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Real = long double;
struct Square { Real x,y,a; int sid; };
struct Result { bool ok=false; Real L=0; std::string reason; };
Real number(const std::string& s) {
    size_t used=0;
    Real x=std::stold(s,&used);
    if(used!=s.size() || !std::isfinite(x)) throw std::runtime_error("invalid finite literal");
    return x;
}
int integer(const std::string& s) {
    size_t used=0; int x=std::stoi(s,&used);
    if(used!=s.size()) throw std::runtime_error("invalid integer");
    return x;
}
std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v; std::stringstream ss(s); std::string x;
    while(std::getline(ss,x,',')) v.push_back(x);
    return v;
}
Result verify(int n,const std::vector<Square>& q) {
    if(int(q.size())!=n) return {false,0,"exact count mismatch"};
    std::set<int> ids;
    for(auto s:q) {
        if(s.sid<0 || s.sid>=n || !ids.insert(s.sid).second) return {false,0,"invalid square ids"};
        if(std::abs(s.x)>10000 || std::abs(s.y)>10000 || std::abs(s.a)>100)
            return {false,0,"input bounds"};
    }
    using Vec=std::array<Real,2>;
    std::vector<Vec> u,v;
    Real xmin=std::numeric_limits<Real>::infinity(),ymin=xmin,xmax=-xmin,ymax=-xmin;
    for(auto s:q) {
        Real c=std::cos(s.a),t=std::sin(s.a);
        u.push_back({c,t}); v.push_back({-t,c});
        Real h=(std::abs(c)+std::abs(t))/2;
        xmin=std::min(xmin,s.x-h); xmax=std::max(xmax,s.x+h);
        ymin=std::min(ymin,s.y-h); ymax=std::max(ymax,s.y+h);
    }
    auto dot=[](Vec a,Vec b){return a[0]*b[0]+a[1]*b[1];};
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) {
        bool separated=false;
        Vec d={q[j].x-q[i].x,q[j].y-q[i].y};
        for(Vec axis:std::array<Vec,4>{u[i],v[i],u[j],v[j]}) {
            Real hi=(std::abs(dot(axis,u[i]))+std::abs(dot(axis,v[i])))/2;
            Real hj=(std::abs(dot(axis,u[j]))+std::abs(dot(axis,v[j])))/2;
            if(std::abs(dot(d,axis))-hi-hj>=0) separated=true;
        }
        if(!separated) return {false,0,"strict allpair overlap"};
    }
    Real L=std::max(xmax-xmin,ymax-ymin);
    if(!std::isfinite(L) || L<=0) return {false,0,"invalid derived bounds"};
    return {true,L,"ok"};
}
int main(int argc,char**argv) {
    try {
        if(argc<2) throw std::runtime_error("CSV required");
        int maxn=1000; std::string output;
        for(int i=2;i<argc;i++) {
            std::string a=argv[i];
            if(a=="--max-n" && i+1<argc) maxn=integer(argv[++i]);
            else if(a=="--details" && i+1<argc) output=argv[++i];
            else if(a=="--threads" && i+1<argc) ++i;
            else throw std::runtime_error("unsupported flag");
        }
        if(maxn<1 || maxn>1000 || output.empty()) throw std::runtime_error("invalid CLI bounds");
        std::ifstream f(argv[1]); if(!f) throw std::runtime_error("cannot open CSV");
        std::string line; std::getline(f,line);
        if(!line.empty() && line.back()=='\r') line.pop_back();
        if(line!="id,n,square_id,x,y,angle") throw std::runtime_error("canonical CSV header required");
        std::map<int,std::vector<Square>> groups;
        int rowid=0;
        while(std::getline(f,line)) {
            if(!line.empty() && line.back()=='\r') line.pop_back();
            auto r=split(line); if(r.size()!=6) throw std::runtime_error("CSV arity");
            if(integer(r[0])!=rowid++) throw std::runtime_error("row id mismatch");
            int n=integer(r[1]);
            if(n<1 || n>1000) throw std::runtime_error("n out of range");
            Square s={number(r[3]),number(r[4]),number(r[5]),integer(r[2])};
            groups[n].push_back(s);
        }
        std::ofstream o(output); if(!o) throw std::runtime_error("cannot write details");
        o<<std::setprecision(std::numeric_limits<Real>::max_digits10);
        o<<"{\"independent_allpairs\":true,\"results\":[";
        bool first=true; int rejected=0;
        for(auto& [n,q]:groups) {
            if(n>maxn) continue;
            Result r=verify(n,q); if(!r.ok) rejected++;
            if(!first) o<<",";
            first=false;
            o<<"{\"n\":"<<n<<",\"L\":";
            if(r.ok)o<<"\""<<r.L<<"\""; else o<<"null";
            o<<",\"reason\":\""<<r.reason<<"\"}";
        }
        o<<"]}\n";
        std::cout<<"{\"rejected\":"<<rejected<<"}\n";
        return 0; // Per-n results, not process success, are the validity gate.
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 2; }
}
