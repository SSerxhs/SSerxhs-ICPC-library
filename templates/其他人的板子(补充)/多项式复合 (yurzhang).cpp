#pragma GCC optimize("Ofast,inline")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,sse4.1,sse4.2,popcnt,abm,mmx,avx,avx2,tune=native")
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

#define MOD 998244353
#define G 332748118
#define N 262210
#define re register
#define gc pa==pb&&(pb=(pa=buf)+fread(buf,1,100000,stdin),pa==pb)?EOF:*pa++
typedef long long ll;
static char buf[100000],*pa(buf),*pb(buf);
static char pbuf[3000000],*pp(pbuf),st[15];
int read() {
    re int x(0);re char c(gc);
    while(c<'0'||c>'9')c=gc;
    while(c>='0'&&c<='9')
        x=x*10+c-48,c=gc;
    return x;
}
void write(re int v) {
    if(v==0)
        *pp++=48;
    else {
        re int tp(0);
        while(v)
            st[++tp]=v%10+48,v/=10;
        while(tp)
            *pp++=st[tp--];
    }
    *pp++=32;
}

int pow(re int a,re int b) {
    re int ans(1);
    while(b)
        ans=b&1?(ll)ans*a%MOD:ans,a=(ll)a*a%MOD,b>>=1;
    return ans;
}

int inv[N],ifac[N];
void pre(re int n) {
    inv[1]=ifac[0]=1;
    for(re int i(2);i<=n;++i)
        inv[i]=(ll)(MOD-MOD/i)*inv[MOD%i]%MOD;
    for(re int i(1);i<=n;++i)
        ifac[i]=(ll)ifac[i-1]*inv[i]%MOD;
}

int getLen(re int t) {
	return 1<<(32-__builtin_clz(t));
}

int lmt(1),r[N],w[N];
void init(re int n) {
	re int l(0);
	while(lmt<=n)
		lmt<<=1,++l;
	for(re int i(1);i<lmt;++i)
		r[i]=(r[i>>1]>>1)|((i&1)<<(l-1));
	re int wn(pow(3,(MOD-1)/lmt));
	w[lmt>>1]=1;
	for(re int i((lmt>>1)+1);i<lmt;++i)
		w[i]=(ll)w[i-1]*wn%MOD;
	for(re int i((lmt>>1)-1);i;--i)
		w[i]=w[i<<1];
}

void DFT(int*a,re int l) {
	static unsigned long long tmp[N];
	re int u(__builtin_ctz(lmt)-__builtin_ctz(l)),t;
	for(re int i(0);i<l;++i)
		tmp[i]=(a[r[i]>>u])%MOD;
	for(re int i(1);i<l;i<<=1)
		for(re int j(0),step(i<<1);j<l;j+=step)
			for(re int k(0);k<i;++k)
				t=(ll)w[i+k]*tmp[i+j+k]%MOD,
				tmp[i+j+k]=tmp[j+k]+MOD-t,
				tmp[j+k]+=t;
	for(re int i(0);i<l;++i)
		a[i]=tmp[i]%MOD;
}

void IDFT(int*a,re int l) {
	std::reverse(a+1,a+l);DFT(a,l);
	re int bk(MOD-(MOD-1)/l);
	for(re int i(0);i<l;++i)
		a[i]=(ll)a[i]*bk%MOD;
}

int n,m;
int a[N],b[N],c[N];

void getInv(int*a,int*b,int deg) {
    if(deg==1)
        b[0]=pow(a[0],MOD-2);
    else {
        static int tmp[N];
        getInv(a,b,(deg+1)>>1);
        re int l(getLen(deg<<1));
        for(re int i(0);i<l;++i)
            tmp[i]=i<deg?a[i]:0;
        DFT(tmp,l),DFT(b,l);
        for(re int i(0);i<l;++i)
            b[i]=(2ll-(ll)tmp[i]*b[i]%MOD+MOD)%MOD*b[i]%MOD;
        IDFT(b,l);
        for(re int i(deg);i<l;++i)
            b[i]=0;
    }
}

void getDer(int*a,int*b,int deg) {
    for(re int i(0);i+1<deg;++i)
        b[i]=(ll)a[i+1]*(i+1)%MOD;
    b[deg-1]=0;
}

void getComp(int*a,int*b,int k,int m,int&n,int*c,int*d) {
    if(k==1) {
        for(re int i(0);i<m;++i)
            c[i]=0,d[i]=b[i];
        n=m,c[0]=a[0];
    } else {
        static int t1[N],t2[N];
        int nl(n),nr(n),*cl,*cr,*dl,*dr;
        getComp(a,b,k>>1,m,nl,cl=c,dl=d);
        getComp(a+(k>>1),b,(k+1)>>1,m,nr,cr=c+nl,dr=d+nl);
        n=std::min(n,nl+nr-1);
        re int _l(getLen(nl+nr));
        for(re int i(0);i<_l;++i)
            t1[i]=i<nl?dl[i]:0;
        for(re int i(0);i<_l;++i)
            t2[i]=i<nr?cr[i]:0;
        DFT(t1,_l),DFT(t2,_l);
        for(re int i(0);i<_l;++i)
            t2[i]=(ll)t1[i]*t2[i]%MOD;
        IDFT(t2,_l);
        for(re int i(0);i<n;++i)
            c[i]=((i<nl?cl[i]:0)+t2[i])%MOD;
        for(re int i(0);i<_l;++i)
            t2[i]=i<nr?dr[i]:0;
        DFT(t2,_l);
        for(re int i(0);i<_l;++i)
            t2[i]=(ll)t1[i]*t2[i]%MOD;
        IDFT(t2,_l);
        for(re int i(0);i<n;++i)
            d[i]=t2[i];
    }
}

void getComp(int*a,int*b,int*c,int deg) {
    static int ts[N],ps[N],c0[N],_t1[N],idM[N];
    int M(std::max((int)ceil(sqrt(deg/log2(deg))*2.5),2)),_n(deg+deg/M);
    getComp(a,b,deg,M,_n,c0,_t1);
    re int _l(getLen(_n+deg));
    for(re int i(_n);i<_l;++i)
        c0[i]=0;
    for(re int i(0);i<_l;++i)
        ps[i]=i==0;
    for(re int i(0);i<_l;++i)
        ts[i]=M<=i&&i<deg?b[i]:0;
    getDer(b,_t1,M);
    for(re int i(M-1);i<deg;++i)
        _t1[i]=0; /// Important!!!
    getInv(_t1,idM,deg);
    for(int i=deg;i<_l;++i)
    	idM[i]=0;
    DFT(ts,_l),DFT(idM,_l);
    for(re int t(0);t*M<deg;++t) {
        for(re int i(0);i<_l;++i)
            _t1[i]=i<deg?c0[i]:0;
        DFT(ps,_l),DFT(_t1,_l);
        for(re int i(0);i<_l;++i)
            _t1[i]=(ll)_t1[i]*ps[i]%MOD,
            ps[i]=(ll)ps[i]*ts[i]%MOD;
        IDFT(ps,_l),IDFT(_t1,_l);
        for(re int i(deg);i<_l;++i)
            ps[i]=0;
        for(re int i(0);i<deg;++i)
            c[i]=((ll)_t1[i]*ifac[t]+c[i])%MOD;
        getDer(c0,c0,_n);
        for(re int i(_n-1);i<_l;++i)
            c0[i]=0;
        DFT(c0,_l);
        for(re int i(0);i<_l;++i)
            c0[i]=(ll)c0[i]*idM[i]%MOD;
        IDFT(c0,_l);
        for(re int i(_n-1);i<_l;++i)
            c0[i]=0;
    }
}

int main() {
    n=read(),m=read();
    for(re int i(0);i<=n;++i)
        a[i]=read();
    for(re int i(0);i<=m;++i)
        b[i]=read();
    
    m=(n>m?n:m)+1;
    pre(m);init(m*5);
    getComp(a,b,c,m);
    
    for(re int i(0);i<=n;++i)
        write(c[i]);
    fwrite(pbuf,1,pp-pbuf,stdout);
    return 0;
}
