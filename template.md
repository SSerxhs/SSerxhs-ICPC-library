<div align='center' ><font size='7'><B>SSerxhs 的 ICPC 板子</B></font></div>

## Content

[TOC]

<div style="page-break-after:always"></div>

version: 2.3.1

###    数据结构 

#### 哈希表

```cpp
template<int N,typename T,typename TT> struct ht//个数，定义域，值域
{
	const static int p=1e6+7,M=p+2;
	TT a[N];
	T v[N];
	int fir[p+2],nxt[N],st[p+2];//和模数相适应
	int tp,ds;//自定义模数
	ht(){memset(fir,0,sizeof fir);tp=ds=0;}
	void mdf(T x,TT z)//位置，值
	{
		ui y=x%p;
		for (int i=fir[y];i;i=nxt[i]) if (v[i]==x) return a[i]=z,void();//若不可能重复不需要 for
		v[++ds]=x;a[ds]=z;
		if (!fir[y]) st[++tp]=y;
		nxt[ds]=fir[y];fir[y]=ds;
	}
	TT find(T x)
	{
		ui y=x%p;
		int i;
		for (i=fir[y];i;i=nxt[i]) if (v[i]==x) return a[i];
		return 0;//返回值和是否判断依据要求决定
	}
	void clear()
	{
		++tp;
		while (--tp) fir[st[tp]]=0;
		ds=0;
	}
};
```

#### 珂朵莉树

```cpp
struct node
{
	int l,r;
	mutable int v;
	node(int L=0,int R=0,int V=0):l(L),r(R),v(V){}
	bool operator<(const node& x) const {return l<x.l;}
};
template<typename T> struct set
{

}
set<node> s;
typedef set<node>::iterator iter;
iter split(int x)
{
	iter it=s.lower_bound(node(x));
	if (it!=s.end()&&it->l==x) return it;
	node t=*--it;s.erase(it);
	s.insert(node(t.l,x-1,t.v));
	return s.insert(node(x,t.r,t.v)).first;
}
rt=split(r+1);lt=split(l);//[lr,rt)
```

#### 可持久化数组

$O((n+q)\log(n))$，$O((n+q)\log (n))$。

```cpp
struct arr
{
	int c[M][2],rt[O],s[M],b[N];
	int ds,n,ver,v,p,i;
	void build(int &x,int l,int r)
	{
		x=++ds;
		if (l==r) {s[x]=b[l];return;}
		build(c[x][0],l,l+r>>1);
		build(c[x][1],(l+r>>1)+1,r);
	}
	void rebuild(int &x,int pre)
	{
		x=++ds;int l=1,r=n,mid,now=x;
		while (l<r)
		{
			mid=l+r>>1;
			if (mid>=p){c[now][1]=c[pre][1];now=c[now][0]=++ds;r=mid;pre=c[pre][0];} else {c[now][0]=c[pre][0];now=c[now][1]=++ds;l=mid+1;pre=c[pre][1];}
		}
		s[now]=v;
	}
	void init(int *a,int nn)
	{
		n=nn;
		for (i=1;i<=n;i++) b[i]=a[i];
		build(rt[0],1,n);
	}
	int mdf(int pv,int pos,int val)
	{
		p=pos,v=val;
		rebuild(rt[++ver],rt[pv]);
		return ver;
	}
	int ask(int ve,int pos)
	{
		int l=1,r=n,x=rt[ve],mid;
		rt[++ver]=rt[ve];
		while (l<r)
		{
			mid=l+r>>1;
			if (mid>=pos) {x=c[x][0];r=mid;} else {x=c[x][1];l=mid+1;}
		}
		return s[x];
	}
};
```

#### 左偏树/可并堆

$O((n+q)\log n)$，$O(n)$。

```cpp
struct left_tree//小根堆，大根堆需要改的地方注释了
{
	int jl[N],v[N],f[N],c[N][2],tf[N],n;//tf只有删非堆顶才用
	bool ed[N];
	void init(const int nn,const int *a)
	{
		jl[0]=-1;n=nn;
		memset(jl+1,0,n<<2);
		memset(tf+1,0,n<<2);//同上
		memset(c+1,0,n<<3);
		memset(ed+1,0,n);
		for (int i=1;i<=n;i++) v[f[i]=i]=a[i];
	}
	int mg(int x,int y)
	{
		if (!(x&&y)) return x|y;
		if (v[x]>v[y]||v[x]==v[y]&&x>y) swap(x,y);//改
		tf[c[x][1]=mg(c[x][1],y)]=x;//同上
		if (jl[c[x][0]]<jl[c[x][1]]) swap(c[x][0],c[x][1]);
		jl[x]=jl[c[x][1]]+1;
		return x;
	}
	int getf(int x)
	{
		if (f[x]==x) return x;
		return f[x]=getf(f[x]);
	}
	int merge(int x,int y)
	{
		if (ed[x]||ed[y]||(x=getf(x))==(y=getf(y))) return x;
		int z=mg(x,y);return f[x]=f[y]=z;
	}
	int getv(int x)//需要自行判断是否存在
	{
		return v[getf(x)];
	}
	int del(int x)//删除堆内最值
	{
		tf[c[x][0]]=tf[c[x][1]]=0;
		f[c[x][0]]=f[c[x][1]]=f[x]=mg(c[x][0],c[x][1]);
		ed[x]=1;c[x][0]=c[x][1]=tf[x]=0;return f[x];
	}
	int del_all(int x)//删除堆内非最值（没验证过）
	{
		int fa=tf[x];
		if (f[c[x][0]]==x) f[c[x][0]]=getf(tf[x]);
		if (f[c[x][1]]==x) f[c[x][1]]=f[tf[x]];
		tf[x]=tf[c[x][0]]=tf[c[x][1]]=0;
		tf[c[fa][c[fa][1]==x]=mg(c[x][0],c[x][1])]=fa;
		c[x][0]=c[x][1]=0;
		while (jl[c[fa][0]]<jl[c[fa][1]])
		{
			swap(c[fa][0],c[fa][1]);
			jl[fa]=jl[c[fa][1]]+1;
			fa=tf[fa];
		}
	}
	void out(int n)
	{
		for (int i=1;i<=n;i++) printf("%d: c%d&%d f%d v%d\n",i,c[i][0],c[i][1],f[i],v[i]);
	}
};
```

#### 树状数组区间修改区间求和

$O(n)\sim O(q\log n)$，$O(n)$。

```cpp
struct bit
{
	ll a[N],b[N],s[N];//有初始值
	int n;
	void init(int nn,int *a)//初始值
	{
		n=nn;s[0]=0;
		for (int i=1;i<=n;i++) s[i]=s[i-1]+a[i];
	}
	void mdf(int l,int r,int dt)
	{
		int i;++r;
		ll j;
		for (i=l;i<=n;i+=i&-i) a[i]+=dt;j=(ll)dt*l;
		for (i=l;i<=n;i+=i&-i) b[i]+=j;
		for (i=r;i<=n;i+=i&-i) a[i]-=dt;j=(ll)dt*r;
		for (i=r;i<=n;i+=i&-i) b[i]-=j;
	}
	ll presum(int x)
	{
		ll r=0;
		for (int i=x;i;i-=i&-i) r+=a[i];r*=(x+1);
		for (int i=x;i;i-=i&-i) r-=b[i];
		return r+s[x];
	}
	ll sum(int l,int r)
	{
		return presum(r)-presum(l-1);
	}
};
```

#### 二维树状数组矩形修改矩形求和

$O(n^2)\sim O(q\log^2n)$，$O(n^2)$

```cpp
struct bit2
{
	ll a[2050][2050],b[2050][2050],c[2050][2050],d[2050][2050];
	int n,m;
	private:
	void cha(ll a[][2050],int x,int y,int z)
	{
		int i,j;
		for (i=x;i<=n;i+=(i&(-i))) for (j=y;j<=m;j+=(j&(-j))) a[i][j]+=z;
	}
	ll he(int x,int y)
	{
		if ((x<=0)||(y<=0)) return 0;
		int i,j;
		ll z=0,w=0;
		for (i=x;i;i-=(i&(-i))) for (j=y;j;j-=(j&(-j))) z+=a[i][j];
		z*=(x+1)*(y+1);
		w=0;
		for (i=x;i;i-=(i&(-i))) for (j=y;j;j-=(j&(-j))) w+=b[i][j];
		z-=w*(y+1);
		w=0;
		for (i=x;i;i-=(i&(-i))) for (j=y;j;j-=(j&(-j))) w+=c[i][j];
		z-=w*(x+1);
		for (i=x;i;i-=(i&(-i))) for (j=y;j;j-=(j&(-j))) z+=d[i][j];
		return z;
	}
	public:
	void init(int x,int y)
	{
		n=x;m=y;
	}
	void add(int u,int v,int x,int y,int z)//(x1,y1,x2,y2,dt)
	{
		cha(a,u,v,z);
		cha(b,u,v,u*z);//小心乘爆
		cha(c,u,v,v*z);
		cha(d,u,v,u*v*z);
		++x;++y;
		if (x<=n)
		{
			cha(a,x,v,-z);
			cha(b,x,v,-z*x);
			cha(c,x,v,-z*v);
			cha(d,x,v,-z*x*v);
		}
		if (y<=m)
		{
			cha(a,u,y,-z);
			cha(b,u,y,-z*u);
			cha(c,u,y,-z*y);
			cha(d,u,y,-z*u*y);
			if (x<=n)
			{
				cha(a,x,y,z);
				cha(b,x,y,z*x);
				cha(c,x,y,z*y);
				cha(d,x,y,z*x*y);
			}
		}
	}
	ll sum(int u,int v,int x,int y)//(x1,y1,x2,y2)
	{
		--u;--v;
		return (he(x,y)+he(u,v)-he(u,y)-he(x,v));
	}
};
```

#### 带修莫队（功能：区间数有多少种不同的数字）

$O(n^{\frac {5}{3}})$，$O(n)$。

```cpp
void qs(int l,int r)
{
	int i=l,j=r,a=bel[z[l+r>>1]],b=bel[y[l+r>>1]],c=pre[l+r>>1];
	while (i<=j)
	{
		while ((bel[z[i]]<a)||(bel[z[i]]==a)&&((bel[y[i]]<b)||(bel[y[i]]==b)&&(pre[i]<c))) ++i;
		while ((bel[z[j]]>a)||(bel[z[j]]==a)&&((bel[y[j]]>b)||(bel[y[j]]==b)&&(pre[j]>c))) --j;
		if (i<=j)
		{
			swap(z[i],z[j]);
			swap(wz[i],wz[j]);
			swap(y[i],y[j]);
			swap(pre[i++],pre[j--]);
		}
	}
	if (i<r) qs(i,r);
	if (l<j) qs(l,j);
}
ksiz=int(pow(n,2.0/3));
void add(int x)
{
	if (++app[col[x]]==1) ++sum;
}
void del(int x)
{
	if (--app[col[x]]==0) --sum;
}
void work(int x)
{
	if ((l<=pos[x])&&(r>=pos[x]))
	{
		del(pos[x]);
		if (++app[ct[x]]==1) ++sum;
	}
	swap(ct[x],col[pos[x]]);
}
		while (l<z[i]) del(l++);
		while (l>z[i]) add(--l);
		while (r<y[i]) add(++r);
		while (r>y[i]) del(r--);
		while (t<pre[i]) work(++t);
		while (t>pre[i]) work(t--);
```

#### 二次离线莫队

$O(n\sqrt n)$，$O(n)$。

珂朵莉给了你一个序列 $a$，每次查询给一个区间 $[l,r]$，查询 $l \leq i< j \leq r$，且 $a_i \oplus a_j$ 的二进制表示下有 $k$ 个 $1$ 的二元组 $(i,j)$ 的个数。$\oplus$ 是指按位异或。

二次离线莫队，通过扫描线，再次将更新答案的过程离线处理，降低时间复杂度。假设更新答案的复杂度为 $O(k)$，它将莫队的复杂度从 $O(nk\sqrt n)$ 降到了 $O(nk + n\sqrt n)$，大大简化了计算。
设 $x$ 对区间 $[l,r]$ 的贡献为 $f(x,[l,r])$，我们考虑区间端点变化对答案的影响：以 $[l..r]$ 变成 $[l..(r+k)]$ 为例，$\forall x \in [r+1,r+k]$ 求 $f(x,[l,x-1])$。
我们可以进行差分：$f(x,[l,x-1])=f(x,[1,x-1])-f(x,[1,l-1])$，这样转化为了一个数对一个前缀的贡献。保存下来所有这样的询问，从左到右扫描数组计算就可以了。  
但是这样做，空间是 $O(n\sqrt n)$ 的，不太优秀，而且时间常数巨大。。
这样的贡献分为两类：

1. 减号左边的贡献永远是一个前缀 和它后面一个数的贡献。这可以预处理出来。
2. 减号右边的贡献对于一次移动中所有的 $x$ 来说，都是不变的。我们打标记的时候，可以只标记左右端点。

这样，减小时间常数的同时，空间降为了 $O(n)$ 级别。是一个很优秀的算法了。处理前缀询问的时候，我们利用异或运算的交换律，即 $a~\mathrm{xor}~b=c \Longleftrightarrow a~\mathrm{xor}~c=b$
开一个桶 $t$，$t[i]$ 表示当前前缀中与 $i$ 异或有 $k$ 个数位为 $1$ 的数有多少个。
则每加入一个数 $a[i]$，对于所有 $\mathrm{popcount}(x)=k$ 的 $x$，$t[a[i]\operatorname{xor}x]\gets t[a[i]\operatorname{xor}x]+1$ 即可。

```cpp
typedef long long ll;
const int N=1e5+2,M=1<<14;
ll f[N],ans[N],ta[N];
int a[N],cnt[M],bel[N],pc[M],st[N];
int n,m,ksiz;
struct Q
{
	int z,y,wz;
	bool operator<(const Q& x) const {return (bel[z]<bel[x.z])||(bel[z]==bel[x.z])&&((y<x.y)&&(bel[z]&1)||(y>x.y)&&(1^bel[z]&1));}
};
Q mq(const int x,const int y,const int z)
{
	Q a;
	a.z=x;a.y=y;a.wz=z;
	return a; 
}
Q q[N];
vector<Q> b[N];
void read(int &x)
{
	int c=getchar();
	while ((c<48)||(c>57)) c=getchar();
	x=c^48;c=getchar();
	while ((c>=48)&&(c<=57))
	{
		x=x*10+(c^48);
		c=getchar();
	}
}
int main()
{
	int i,j,k,l=1,r=0,tp=0,x,na;
	read(n);read(m);read(k);ksiz=sqrt(n);
	for (i=1;i<=n;i++) {read(a[i]);bel[i]=(i-1)/ksiz+1;}
	if (k==0) st[++tp]=0;
	for (i=1;i<16384;i++)
	{
		if (i&1) pc[i]=pc[i>>1]+1; else pc[i]=pc[i>>1];
		if (pc[i]==k) st[++tp]=i;
	}
	for (i=1;i<=n;i++)
	{
		j=tp+1;f[i]=f[i-1];
		while (--j) f[i]+=cnt[st[j]^a[i]];
		++cnt[a[i]];
	}
	for (i=1;i<=m;i++) {read(q[i].z);read(q[q[i].wz=i].y);}
	sort(q+1,q+m+1);
	for (i=1;i<=m;i++)
	{
		ans[i]=f[q[i].y]-f[r]+f[q[i].z-1]-f[l-1];
		if (k==0) ans[i]+=q[i].z-l;
		if (r<q[i].y)
		{
			b[l-1].push_back(mq(r+1,q[i].y,-i));
			r=q[i].y;
		}
		if (l>q[i].z)
		{
			b[r].push_back(mq(q[i].z,l-1,i));
			l=q[i].z;
		}
		if (r>q[i].y)
		{
			b[l-1].push_back(mq(q[i].y+1,r,i));
			r=q[i].y;
		}
		if (l<q[i].z)
		{
			b[r].push_back(mq(l,q[i].z-1,-i));
			l=q[i].z;
		}
	}
	memset(cnt,0,sizeof(cnt));
	for (i=1;i<=n;i++)
	{
		j=tp+1;x=a[i];
		while (--j) ++cnt[x^st[j]];
		for (j=0;j<b[i].size();j++)
		{
			na=0;l=b[i][j].z;r=b[i][j].y;
			for (k=l;k<=r;k++) na+=cnt[a[k]];
			if (b[i][j].wz>0) ans[b[i][j].wz]+=na; else ans[-b[i][j].wz]-=na;
		}
	}
	for (i=2;i<=m;i++) ans[i]+=ans[i-1];
	for (i=1;i<=m;i++) ta[q[i].wz]=ans[i];
	for (i=1;i<=m;i++) printf("%lld\n",ta[i]);
}
```

#### 回滚莫队

$O(n\sqrt n)$，$O(n)$。

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=2e5+2;
int a[N],z[N],y[N],wz[N],b[N],d[N],bel[N],ans[N],st[N][2],pos[N][2];
int n,m,i,j,x,c,ksiz,gs,l=1,r,tp,na,ca;
void read(int &x)
{
	c=getchar();
	while ((c<48)||(c>57)) c=getchar();
	x=c^48;c=getchar();
	while ((c>=48)&&(c<=57))
	{
		x=x*10+(c^48);
		c=getchar();
	}
}
void qs(int l,int r)
{
	int i=l,j=r,m=bel[z[l+r>>1]],mm=y[l+r>>1];
	while (i<=j)
	{
		while ((bel[z[i]]<m)||(bel[z[i]]==m)&&(y[i]<mm)) ++i;
		while ((bel[z[j]]>m)||(bel[z[j]]==m)&&(y[j]>mm)) --j;
		if (i<=j)
		{
			swap(wz[i],wz[j]);
			swap(z[i],z[j]);
			swap(y[i++],y[j--]);
		}
	}
	if (i<r) qs(i,r);
	if (l<j) qs(l,j);
}
int main()
{
	read(n);ksiz=sqrt(n);
	for (i=1;i<=n;i++) {read(a[i]);b[i]=a[i];bel[i]=(i-1)/ksiz+1;}
	sort(b+1,b+n+1);
	d[gs=1]=b[1];
	for (i=2;i<=n;i++) if (b[i]!=b[i-1]) d[++gs]=b[i];
	for (i=1;i<=n;i++) a[i]=lower_bound(d+1,d+gs+1,a[i])-d;
	read(m);assert(int(n/sqrt(m)));
	for (i=1;i<=m;i++) {read(z[i]);read(y[wz[i]=i]);}
	qs(1,m);
	for (i=1;i<=m;i++)
	{
		if (bel[z[i]]>bel[z[i-1]])
		{
			while (l<=r) {pos[a[l]][0]=pos[a[l]][1]=0;++l;}na=0;
			if (bel[z[i]]==bel[y[i]])
			{
				for (j=z[i];j<=y[i];j++) if (pos[a[j]][0]) na=max(na,j-pos[a[j]][0]); else pos[a[j]][0]=j;
				ans[wz[i]]=na;for (j=z[i];j<=y[i];j++) pos[a[j]][0]=0;na=0;l=ksiz*bel[z[i]];r=l-1;
				continue;
			}
			l=ksiz*bel[z[i]];r=l-1;na=0;
		}
		if (bel[z[i]]==bel[y[i]])
		{
			while (l<=r) {pos[a[l]][0]=pos[a[l]][1]=0;++l;}na=0;
			for (j=z[i];j<=y[i];j++) if (pos[a[j]][0]) na=max(na,j-pos[a[j]][0]); else pos[a[j]][0]=j;
			ans[wz[i]]=na;for (j=z[i];j<=y[i];j++) pos[a[j]][0]=0;
			l=ksiz*bel[z[i]];r=l-1;na=0;
			continue;
		}
		while (r<y[i])
		{
			x=a[++r];pos[x][1]=r;
			if (!pos[x][0]) pos[x][0]=r; else na=max(na,r-pos[x][0]);
		}c=na;
		while (l>z[i])
		{
			x=a[--l];st[++tp][0]=x;st[tp][1]=pos[x][0];
			pos[x][0]=l;
			if (!pos[x][1])
			{
				st[++tp][0]=x+n;st[tp][1]=0;
				pos[x][1]=l;
			} else na=max(na,pos[x][1]-l);
		}
		ans[wz[i]]=na;na=c;++tp;l=ksiz*bel[z[i]];
		while (--tp) if (st[tp][0]<=n) pos[st[tp][0]][0]=st[tp][1]; else pos[st[tp][0]-n][1]=st[tp][1];
	}
	for (i=1;i<=m;i++) printf("%d\n",ans[i]);
}
```

#### splay

$O(n)$，$O((n+q)\log n)$。

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned int ui;
const int N=1e6+20,p=998244353;
void inc(int &x,const int y){if ((x+=y)>=p) x-=p;}
void dec(int &x,const int y){if ((x-=y)<0) x+=p;}
void mul(int &x,const int y){x=(ll)x*y%p;}
template<int N> struct _splay
{
	int c[N][2],plz[N],clz[N],st[N],siz[N],s[N],v[N],f[N];
	bool fg[N],flz[N];
	int tp,rt;
	void allout(int x)
	{
		if (!x) return;
		pushdown(x);
		allout(c[x][0]);
		if (x>2) printf("%d ",v[x]);
		allout(c[x][1]);
	}
	void out(int x)
	{
		printf("%d: c %d & %d f %d s %d v %d siz %d\n",x,c[x][0],c[x][1],f[x],s[x],v[x],siz[x]);
		if (c[x][0]) out(c[x][0]);
		if (c[x][1]) out(c[x][1]);
		if (x==rt) puts("-------------------");
	}
	void iinit()
	{
		for (int i=1;i<N;i++) st[N-i]=i;
		tp=N-1;
	}
	void init()
	{
		tp=N-3;
		c[1][0]=c[1][1]=flz[1]=plz[1]=fg[1]=v[1]=f[1]=s[1]=0;clz[1]=1;
		c[2][0]=c[2][1]=flz[2]=plz[2]=fg[2]=v[2]=f[2]=s[2]=0;clz[2]=1;
		c[1][1]=2;f[2]=1;rt=1;siz[2]=1;siz[1]=2;
	}
	void pushup(int x)
	{
		s[x]=((ui)s[c[x][0]]+s[c[x][1]]+v[x])%p;
		siz[x]=siz[c[x][0]]+siz[c[x][1]]+1;
	}
	void pushdown(int x)
	{
		int lc=c[x][0],rc=c[x][1];
		if (flz[x])
		{
			if (lc) flz[lc]^=1,swap(c[lc][0],c[lc][1]);
			if (rc) flz[rc]^=1,swap(c[rc][0],c[rc][1]);
			flz[x]=0;
		}
		if (fg[x])
		{
			clz[x]=1;plz[x]=0;
			if (lc) fg[lc]=1,v[lc]=v[x],s[lc]=(ll)v[x]*siz[lc]%p;
			if (rc) fg[rc]=1,v[rc]=v[x],s[rc]=(ll)v[x]*siz[rc]%p;
			fg[x]=0;
		}
		else
		{
			if (clz[x]!=1)
			{
				if (lc) mul(clz[lc],clz[x]),mul(s[lc],clz[x]),mul(plz[lc],clz[x]),mul(v[lc],clz[x]);
				if (rc) mul(clz[rc],clz[x]),mul(s[rc],clz[x]),mul(plz[rc],clz[x]),mul(v[rc],clz[x]);
				clz[x]=1;
			}
			if (plz[x])
			{
				if (lc) inc(plz[lc],plz[x]),inc(v[lc],plz[x]),s[lc]=(s[lc]+(ll)siz[lc]*plz[x])%p;
				if (rc) inc(plz[rc],plz[x]),inc(v[rc],plz[x]),s[rc]=(s[rc]+(ll)siz[rc]*plz[x])%p;
				plz[x]=0;
			}
		}
	}
	void zigzag(int x)
	{
		int y=f[x],z=f[y],typ=(c[y][0]==x);
		if (z) c[z][c[z][1]==y]=x;
		f[x]=z;f[y]=x;c[y][typ^1]=c[x][typ];
		if (c[x][typ]) f[c[x][typ]]=y;
		c[x][typ]=y;
		pushup(y);
	}
	void allpd(int x)
	{
		static int st[N],tp;
		st[tp=1]=x;
		while (x=f[x]) st[++tp]=x;
		while (tp) pushdown(st[tp--]);
	}
	void splay(int x,int tar)
	{
		if (!tar) rt=x;
		int y;
		while ((y=f[x])!=tar)
		{
			if (f[y]!=tar) zigzag(c[f[y]][0]==y^c[y][0]==x?x:y);
			zigzag(x);
		}
		pushup(x);
	}
	void find(int kth,int tar)
	{
		int x=rt;
		while (siz[c[x][0]]+1!=kth)
		{
			pushdown(x);
			if (siz[c[x][0]]>=kth) x=c[x][0]; else
			{
				kth-=siz[c[x][0]]+1;
				x=c[x][1];
			}
		}
		pushdown(x);
		splay(x,tar);
	}
	int rk(int x)
	{
		allpd(x);
		splay(x,0);
		return siz[c[x][0]];
	}
	void split(int x,int y)
	{
		find(x,0);find(y+2,rt);
	}
	int npt()
	{
		int x=st[tp--];
		c[x][0]=c[x][1]=plz[x]=siz[x]=s[x]=v[x]=fg[x]=flz[x]=0;
		clz[x]=1;
		return x;
	}
	int build(int *a,int l,int r)
	{
		if (l>r) return 0;
		int m=l+r>>1,x;
		v[x=npt()]=a[m];
		//printf("build %d %d %d\n",l,r,x);
		if (l==r)
		{
			siz[x]=1;
			s[x]=v[x];
			return x;
		}
		c[x][0]=build(a,l,m-1);
		c[x][1]=build(a,m+1,r);
		if (c[x][0]) f[c[x][0]]=x;
		if (c[x][1]) f[c[x][1]]=x;
		pushup(x);
		return x;
	}
	void ins(int pos,int *a,int n)//在pos后插入
	{
		if (!n) return;
		split(pos+1,pos);
		// out(rt);
		int x=c[rt][1];
		c[x][0]=build(a,1,n);
		// printf("%d %d\n",x,c[x][0]);
		f[c[x][0]]=x;
		pushup(x);pushup(rt);
	}
	void del(int l,int r)//删除[l,r]
	{
		split(l,r);
		c[c[rt][1]][0]=0;
		pushup(c[rt][1]);
		pushup(rt);
	}
	void rev(int l,int r)
	{
		split(l,r);
		int x=c[c[rt][1]][0];
		swap(c[x][0],c[x][1]);
		flz[x]^=1;
	}
	void add(int l,int r,int val)
	{
		split(l,r);
		int x=c[c[rt][1]][0];
		inc(v[x],val);inc(plz[x],val);
		s[x]=(s[x]+(ll)val*siz[x])%p;
		pushup(f[x]);pushup(rt);
	}
	void multi(int l,int r,int val)
	{
		split(l,r);
		int x=c[c[rt][1]][0];
		mul(v[x],val);mul(plz[x],val);
		mul(s[x],val);mul(clz[x],val);
		pushup(f[x]);pushup(rt);
	}
	void mov(int l1,int r1,int l2)//都是原下标
	{
		if (l2>l1) l2-=r1-l1+1;
		split(l1,r1);int x=c[c[rt][1]][0];
		allpd(x);c[f[x]][0]=0;
		pushup(f[x]);pushup(rt);
		split(l2+1,l2);
		allpd(c[rt][1]);
		c[c[rt][1]][0]=x;f[x]=c[rt][1];
		pushup(f[x]);pushup(rt);
	}
	int sum(int l,int r)
	{
		split(l,r);//puts("spe ");out(rt);
		return s[c[c[rt][1]][0]];
	}
};
_splay<N> s;
int a[N];
int n,q,i,x,y,z;
void read(int &x)
{
	int c=getchar();
	while (c<48||c>57) c=getchar();
	x=c^48;c=getchar();
	while (c>=48&&c<=57) x=x*10+(c^48),c=getchar();
}
int main()
{
	read(n);read(q);s.iinit();
	for (i=1;i<=n;i++) a[i]=i;
	s.init();s.ins(0,a,n);//s.out(s.rt);
	while (q--)
	{
		read(x);read(y);s.rev(x,y);
	}
	s.allout(s.rt);
}
```

#### 区间线性基

$O((n+q)\log a)$，$O(n\log a)$。

```cpp
int v[N][M][2];//N是序列总长，M是位数
void ins(int x,int p)//在p插入x
{
	memcpy(v[p],v[p-1],sizeof(v[p]));int n=p,y;
	for (int i=M-1;x;i--) if (x&1<<i)
	{
		if (v[n][i][0])
		{
			if (p>v[n][i][1])
			{
				y=v[n][i][0];v[n][i][0]=x;
				x^=y;swap(p,v[n][i][1]);
			} else x^=v[n][i][0];
		} else v[n][i][0]=x,v[n][i][1]=p,x=0;
	}
}

ans=0;//[z,y]
for (i=M-1;~i;i--) if ((1^ans>>i&1)&&(v[y][i][1]>=z)) ans^=v[y][i][0];
```

#### 第 $k$ 大线性基

$O((n+q)\log a)$，$O(\log a)$。

 ```cpp
 void ins(ll x)
 {
 	if (x==0) return con=1,void();//con=1:有0
 	int i;
 	for (i=50;x;i--) if (x>>i&1)
 	{
 		if (!ji[i]) {ji[i]=x;i=-1;break;}x^=ji[i];
 	}
 	if (!x) con=1;
 }
 ll kmax(ll x)//若有初始值改 r 即可
 {
 	ll r=0;
 	int m=0,i;
 	for (i=50;~i;i--) if (ji[i]) a[++m]=i;
 	if (1ll<<m<=x-con) return -1;//个数少于k
 	x=(1ll<<m)-x;
 	for (i=1;i<=m;i++) if ((x>>m-i^r>>a[i])&1) r^=ji[a[i]];
 	return r;
 }
 ll kmin(ll x)//若有初始值改 r 即可
 {
 	ll r=0;
 	int m=0,i;
 	for (i=50;~i;i--) if (ji[i]) a[++m]=i;
 	x-=con;
 	if (1ll<<m<=x) return -1;//个数少于k
 	for (i=1;i<=m;i++) if ((x>>m-i^r>>a[i])&1) r^=ji[a[i]];
 	return r;
 }
 
 ```

#### fhq-treap

$O((n+q)\log n)$，$O(n)$。

```cpp
const int N=1.1e6+2;
int c[N][2],v[N],w[N],s[N];
int n,i,x,y,ds,val,kth,p,q,z,rt,la,m,ans;
void pushup(const int x)
{
	s[x]=s[c[x][0]]+s[c[x][1]]+1;
}
void split_val(int now,int &x,int &y)//调用外部val,相等归入y
{
	if (!now) return x=y=0,void();
	if (val<=v[now]) split_val(c[y=now][0],x,c[now][0]);
	else split_val(c[x=now][1],c[now][1],y);
	pushup(now);
}
void split_kth(int now,int &x,int &y)//调用外部kth
{
	if (!now) return x=y=0,void();
	if (kth<=s[c[now][0]]) split_kth(c[y=now][0],x,c[now][0]);
	else kth-=s[c[now][0]]+1,split_kth(c[x=now][1],c[now][1],y);
	pushup(now);
}
int merge(int x,int y)//小根ver.
{
	if (!(x&&y)) return x|y;
	if (w[x]<w[y]) {c[x][1]=merge(c[x][1],y);pushup(x);return x;}
	else {c[y][0]=merge(x,c[y][0]);pushup(y);return y;}
}
int main()
{
	read(n);read(m);srand(998244353);
	for (i=1;i<=n;i++)
	{
		read(x);val=v[++ds]=x;w[ds]=rand();s[ds]=1;split_val(rt,p,q);rt=merge(merge(p,ds),q);
	}
	while (m--)
	{
		read(y);read(x);x^=la;
		if (y==4)
		{
			kth=x;split_kth(rt,p,q);x=p;
			while (c[x][1]) x=c[x][1];
			ans^=(la=v[x]);rt=merge(p,q);
			continue;
		}
		val=x;
		if (y==1)
		{
			v[++ds]=x;w[ds]=rand();s[ds]=1;
			split_val(rt,p,q);rt=merge(merge(p,ds),q);
			continue;
		}
		if (y==2)
		{
			split_val(rt,p,q);kth=1;split_kth(q,i,z);
			rt=merge(p,z);continue;
		}
		if (y==3)
		{
			split_val(rt,p,q);ans^=(la=s[p]+1);
			rt=merge(p,q);continue;
		}
		if (y==5)
		{
			split_val(rt,p,q);x=p;
			while (c[x][1]) x=c[x][1];ans^=(la=v[x]);
			rt=merge(p,q);continue;
		}
		++val;split_val(rt,p,q);x=q;
		while (c[x][0]) x=c[x][0];
		ans^=(la=v[x]);rt=merge(p,q);
	}printf("%d",ans);
}
```

#### 笛卡尔树

$O(n)$，$O(n)$。

```cpp
int c[N][2],p[N],st[N];
int main()
{
	int i,n,tp=0;
	ll la=0,ra=0;
	read(n);
	for (i=1;i<=n;i++)
	{
		read(p[i]);st[tp+1]=0;
		while ((tp)&&(p[st[tp]]>p[i])) --tp;
		c[c[st[tp]][1]=i][0]=st[tp+1];st[++tp]=i;
	}
	for (i=1;i<=n;i++) la^=(ll)i*(c[i][0]+1);
	for (i=1;i<=n;i++) ra^=(ll)i*(c[i][1]+1);
	printf("%lld %lld",la,ra);
}
```

#### 扫描线

$O((n+q)\log n)$，$O(n+q)$。

```cpp
const int N=2e5+2,M=8e5+2;//2倍N
struct Q
{
	int l,r,h,typ;
	Q(int a=0,int b=0,int c=0,int d=0):l(a),r(b),h(c),typ(d){}
	bool operator<(const Q &o) const {return h<o.h;}
};
ll ans;
Q q[N];
int l[M],r[M],s[M][2],lz[M];
int xx[N>>1][2],yy[N>>1][2],a[N];
int n,i,j,x,y,z,dt,m,len;
void pushup(int x)
{
	int c=x<<1;
	if (s[c][0]==s[c|1][0]) s[x][0]=s[c][0],s[x][1]=s[c][1]+s[c|1][1];
	else
	{
		if (s[c][0]>s[c|1][0]) c|=1; 
		s[x][0]=s[c][0];s[x][1]=s[c][1];
	}
}
void pushdown(int x)
{
	if (lz[x])
	{
		int c=x<<1;
		lz[c]+=lz[x];s[c][0]+=lz[x];c|=1;
		lz[c]+=lz[x];s[c][0]+=lz[x];lz[x]=0;
	}
}
void build(int x)
{
	if (l[x]==r[x]) return s[x][1]=a[l[x]+1]-a[l[x]],void();
	int c=x<<1;
	l[c]=l[x];r[c]=l[x]+r[x]>>1;
	l[c|1]=r[c]+1;r[c|1]=r[x];
	build(c);build(c|1);
	pushup(x);
}
void mdf(int x)
{
	if (z<=l[x]&&r[x]<=y)
	{
		lz[x]+=dt;s[x][0]+=dt;return;
	}
	pushdown(x);
	int c=x<<1;
	if (z<=r[c]) mdf(c);
	if (y>r[c]) mdf(c|1);
	pushup(x);
}
int main()
{
	read(n);
	for (i=1;i<=n;i++)
	{
		read(xx[i][0]);read(yy[i][0]);
		read(xx[i][1]);read(yy[i][1]);
		a[++m]=xx[i][0],a[++m]=xx[i][1];
	}
sort(a+1,a+m+1);r[l[1]=1]=(m=unique(a+1,a+m+1)-a-1)-1;build(1);
	for (i=1;i<=n;i++)
	{
		xx[i][0]=lower_bound(a+1,a+m+1,xx[i][0])-a;
		xx[i][1]=lower_bound(a+1,a+m+1,xx[i][1])-a;
		q[i]=Q(xx[i][0],xx[i][1],yy[i][0],1);
		q[i+n]=Q(xx[i][0],xx[i][1],yy[i][1],-1);
	}m=n<<1;len=a[m]-a[1];
	sort(q+1,q+m+1);
	for (i=1;i<=m;i++)
	{
		z=q[i].l;y=q[i].r-1;dt=q[i].typ;mdf(1);
		ans+=(ll)(s[1][0]?len:len-s[1][1])*(q[i+1].h-q[i].h);
	}
	printf("%lld\n",ans);
}
```

#### Segmenttree Beats!

$O((n+q)\log n)\sim O(n+q\log^2 n)$，$O(n)$。

- `1 l r k`：对于所有的 $i\in[l,r]$，将 $A_i$ 加上 $k$（$k$ 可以为负数）。
- `2 l r v`：对于所有的 $i\in[l,r]$，将 $A_i$ 变成 $\min(A_i,v)$。
- `3 l r`：求 $\sum_{i=l}^{r}A_i$。
- `4 l r`：对于所有的 $i\in[l,r]$，求 $A_i$ 的最大值。
- `5 l r`：对于所有的 $i\in[l,r]$，求 $B_i$ 的最大值。

```cpp
typedef long long ll;d
struct Q
{
	ll mxp,mx,vp,v;
	Q(ll a=0,ll b=0,ll c=0,ll d=0):mxp(a),mx(b),vp(c),v(d){}
};
const int N=5e5+2,M=2e6+2;
const ll inf=-1e18;
Q lz[M];
ll mx[M],cnt[M],se[M],pmx[M],s[M],ans;
int l[M],r[M],cd[M],a[N];
int n,m,i,x,y,z,typ,dt;
void pushup(int x)
{
	int lc=x<<1,rc=lc|1;
	s[x]=s[lc]+s[rc];
	pmx[x]=max(pmx[lc],pmx[rc]);
	if (mx[lc]==mx[rc])
	{
		mx[x]=mx[lc];cnt[x]=cnt[lc]+cnt[rc];
		se[x]=max(se[lc],se[rc]);
	}
	else if (mx[lc]<mx[rc]) mx[x]=mx[rc],cnt[x]=cnt[rc],se[x]=max(mx[lc],se[rc]);
	else mx[x]=mx[lc],cnt[x]=cnt[lc],se[x]=max(mx[rc],se[lc]);
}
void build(int x)
{
	cd[x]=r[x]-l[x]+1;
	if (l[x]==r[x]) return s[x]=mx[x]=pmx[x]=a[l[x]],se[x]=inf,cnt[x]=1,void();
	int c=x<<1;
	l[c]=l[x];r[c]=l[x]+r[x]>>1;
	l[c|1]=r[c]+1;r[c|1]=r[x];
	build(c);build(c|1);
	pushup(x);
}
void upd(int x,Q o)
{
	lz[x]=Q(max(lz[x].mxp,lz[x].mx+o.mxp),lz[x].mx+o.mx,max(lz[x].vp,lz[x].v+o.vp),lz[x].v+o.v);
	s[x]+=o.mx*cnt[x]+o.v*(cd[x]-cnt[x]);se[x]+=o.v;
	pmx[x]=max(pmx[x],mx[x]+o.mxp);mx[x]+=o.mx;
}
void pushdown(int x)
{
	int c=x<<1;
	ll mxx=max(mx[c],mx[c|1]);
	if (mx[c]==mxx) upd(c,lz[x]); else upd(c,Q(lz[x].vp,lz[x].v,lz[x].vp,lz[x].v));
	c|=1;
	if (mx[c]==mxx) upd(c,lz[x]); else upd(c,Q(lz[x].vp,lz[x].v,lz[x].vp,lz[x].v));
	lz[x]=Q();
}
void mdf1(int x)
{
	if (z<=l[x]&&r[x]<=y)
	{
		upd(x,Q(max(dt,0),dt,max(dt,0),dt));
		return;
	}
	pushdown(x);
	int c=x<<1;
	if (z<=r[c]) mdf1(c);
	if (y>r[c]) mdf1(c|1);
	pushup(x);
}
void mdf2(int x)
{
	if (dt>=mx[x]) return;
	if (z<=l[x]&&r[x]<=y)
	{
		if (dt<=se[x])
		{
			pushdown(x);
			mdf2(x<<1);mdf2(x<<1|1);
			pushup(x);
		}
		else
		{
			upd(x,Q(0,dt-mx[x],0,0));
		}
		return;
	}
	pushdown(x);
	int c=x<<1;
	if (z<=r[c]) mdf2(c);
	if (y>r[c]) mdf2(c|1);
	pushup(x);
}
void sol3(int x)
{
	if (z<=l[x]&&r[x]<=y) return ans+=s[x],void();
	pushdown(x);
	int c=x<<1;
	if (z<=r[c]) sol3(c);
	if (y>r[c]) sol3(c|1);
}
void sol4(int x)
{
	if (z<=l[x]&&r[x]<=y) return ans=max(ans,mx[x]),void();
	pushdown(x);
	int c=x<<1;
	if (z<=r[c]) sol4(c);
	if (y>r[c]) sol4(c|1);
}
void sol5(int x)
{
	if (z<=l[x]&&r[x]<=y) return ans=max(ans,pmx[x]),void();
	pushdown(x);
	int c=x<<1;
	if (z<=r[c]) sol5(c);
	if (y>r[c]) sol5(c|1);
}
int main()
{
	read(n);read(m);
	for (i=1;i<=n;i++) read(a[i]);
	r[l[1]=1]=n;build(1);
	while (m--)
	{
		read(typ);read(z);read(y);
		if (typ>=3)
		{
			ans=(typ==3)?0:inf;
			if (typ==3) sol3(1); else if (typ==4) sol4(1); else sol5(1);
			printf("%lld\n",ans);
		}
		else
		{
			read(dt);
			if (typ==1) mdf1(1); else mdf2(1);
		}
	}
} 
```

#### $k$-d 树（二进制分组）

均摊 $O(\log ^2n)$ 插入，$O(\sqrt n)$ 矩形查询。

```cpp
     
```

<div style="page-break-after:always"></div>

### 数学

#### 矩阵求逆（要求质数）

$O(n^3)$，$O(n^2)$。

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=402,p=1e9+7;
void inv(int &x)
{
	int y=p-2,r=1;
	while (y)
	{
		if (y&1) r=(ll)r*x%p;
		x=(ll)x*x%p;
		y>>=1;
	}
	x=r;
}
int a[N][N],ih[N],jh[N];
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	int i,j,k,n;
	cin>>n;
	for (i=1;i<=n;i++) for (j=1;j<=n;j++) cin>>a[i][j];
	for (k=1;k<=n;k++)
	{//ih,jh要清空
		for (i=k;i<=n;i++) if (!ih[k]) for (j=k;j<=n;j++) if (a[i][j])
		{
			ih[k]=i;jh[k]=j;break;
		}
		if (!ih[k]) return cout<<"No Solution"<<endl,0;
		for (j=1;j<=n;j++) swap(a[k][j],a[ih[k]][j]);
		for (i=1;i<=n;i++) swap(a[i][k],a[i][jh[k]]);
		if (!a[k][k]) return cout<<"No Solution"<<endl,0;inv(a[k][k]);
		for (i=1;i<=n;i++) if (i!=k) a[k][i]=(ll)a[k][i]*a[k][k]%p;
		for (i=1;i<=n;i++) if (i!=k) for (j=1;j<=n;j++) if (j!=k) a[i][j]=(a[i][j]+(ll)(p-a[i][k])*a[k][j])%p;
		for (i=1;i<=n;i++) if (i!=k) a[i][k]=(ll)(p-a[i][k])*a[k][k]%p;	
	}
	for (k=n;k;k--)
	{
		for (j=1;j<=n;j++) swap(a[k][j],a[jh[k]][j]);
		for (i=1;i<=n;i++) swap(a[i][k],a[i][ih[k]]);
	}
	for (i=1;i<=n;i++)
	{
		for (j=1;j<=n;j++) cout<<a[i][j]<<" \n"[j==n];
	}
}
/*
输入
3
1 2 8
2 5 6
5 1 2
输出
718750005 718750005 968750007
171875001 671875005 296875002
117187501 867187506 429687503
*/
```

#### 任意模数矩阵求逆（未验）

$O(n^3)$，$O(n^2)$。

```cpp
int ksm(int x,int y)
{
	int r=1;
	while (y)
	{
		if (y&1) r=(ll)r*x%p;
		y>>=1;x=(ll)x*x%p;
	}
	return r;
}
int phi(int n)
{
	int r=n;
	for (int i=2;i*i<=n;i++) if (n%i==0)
	{
		r=r/i*(i-1);n/=i;
		while (n%i==0) n/=i;
	}
	if (n>1) r=r/n*(n-1);
	return r;
}
void cal(int a[][N],int b[][N],int n)
{
	int i,j,k,r,ph=phi(p);
	for (i=1;i<=n;i++) memset(b+1,0,n<<2);
	for (i=1;i<=n;i++) b[i][i]=1;
	for (i=1;i<=n;i++)
	{
		k=i;
		for (j=i+1;j<=n;j++) if (a[j][i]&&a[j][i]<a[k][i]) k=j;
		if (!a[k][i]) {puts("No Solution");exit(0);}
		swap(a[i],a[k]);swap(b[i],b[k]);
		for (j=i+1;j<=n;j++) if (a[j][i])
		{
			r=p-a[j][i]/a[i][i];
			for (k=i;k<=n;k++) a[j][k]=(a[j][k]+(ll)r*a[i][k])%p;
			for (k=1;k<=n;k++) b[j][k]=(b[j][k]+(ll)r*b[i][k])%p;
			while (a[j][i])
			{
				swap(a[i],a[j]);swap(b[i],b[j]);
				r=p-a[j][i]/a[i][i];
				for (k=i;k<=n;k++) a[j][k]=(a[j][k]+(ll)r*a[i][k])%p;
				for (k=1;k<=n;k++) b[j][k]=(b[j][k]+(ll)r*b[i][k])%p;
			}
		}
		if (__gcd(a[i][i],p)!=1) {puts("No Solution");exit(0);}
		r=ksm(a[i][i],ph-1);
		for (j=i;j<=n;j++) a[i][j]=(ll)a[i][j]*r%p;
		for (j=1;j<=n;j++) b[i][j]=(ll)b[i][j]*r%p;
		assert(a[i][i]==1);
		for (j=1;j<i;j++)
		{
			r=p-a[j][i];
			for (k=i;k<=n;k++) a[j][k]=(a[j][k]+(ll)r*a[i][k])%p;
			for (k=1;k<=n;k++) b[j][k]=(b[j][k]+(ll)r*b[i][k])%p;
		}
	}
}
```

#### 矩阵的特征多项式

$O(n^3)$，$O(n^2)$。

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=502,p=998244353;
int a[N][N],f[N];
int n,i,j,k,x,y,r;
void inc(int &x,const int y)
{
	if ((x+=y)>=p) x-=p;
}
void dec(int &x,const int y)
{
	if ((x-=y)<0) x+=p;
}
int ksm(int x,int y)
{
	int r=1;
	while (y)
	{
		if (y&1) r=(ll)r*x%p;
		x=(ll)x*x%p;y>>=1;
	}
	return r;
}
void calmatrix(int a[N][N],int n)
{
	int i,j,k,r;
	for (i=2;i<=n;i++)
	{
		for (j=i;j<=n&&!a[j][i-1];j++);
		if (j>n) continue;
		if (j>i)
		{
			swap(a[i],a[j]);
			for (k=1;k<=n;k++) swap(a[k][j],a[k][i]);
		}
		r=a[i][i-1];
		for (j=1;j<=n;j++) a[j][i]=(ll)a[j][i]*r%p;
		r=ksm(r,p-2);
		for (j=i-1;j<=n;j++) a[i][j]=(ll)a[i][j]*r%p;
		for (j=i+1;j<=n;j++)
		{
			r=a[j][i-1];
			for (k=1;k<=n;k++) a[k][i]=(a[k][i]+(ll)a[k][j]*r)%p;
			r=p-r;
			for (k=i-1;k<=n;k++) a[j][k]=(a[j][k]+(ll)a[i][k]*r)%p;
		}
	}
}
void calpoly(int a[N][N],int n,int *f)
{
	static int g[N][N];
	memset(g,0,sizeof(g));
	g[0][0]=1;
	int i,j,k,r,rr;
	for (i=1;i<=n;i++)
	{
		r=p-1;
		for (j=i;j;j--)//第 j 行选第 n 列
		{
			rr=(ll)r*a[j][i]%p;
			for (k=0;k<j;k++) g[i][k]=(g[i][k]+(ll)rr*g[j-1][k])%p;
			r=(ll)r*a[j][j-1]%p;
		}
		for (k=1;k<=i;k++) inc(g[i][k],g[i-1][k-1]);
	}
	memcpy(f,g[n],n+1<<2);
	//if (n&1) for (i=0;i<=n;i++) if (f[i]) f[i]=p-f[i];//若注释掉则为 |kE-A|
}
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	cin>>n;
	for (i=1;i<=n;i++) for (j=1;j<=n;j++) cin>>a[i][j];
	calmatrix(a,n);calpoly(a,n,f);
	for (i=0;i<=n;i++) cout<<f[i]<<"x^"<<i<<"+\n"[i==n];
}
/*
3
1 2 3
4 5 6
7 8 9
输出：0x^0+998244335x^1+998244338x^2+1x^3
*/
```

#### Strassen 矩阵乘法

$O(n^{\log_27})$。

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef unsigned int ui;
typedef unsigned long long ull;
const ui p=998244353;
const ull fh=1ull<<31;
struct Q
{
	ui **a;
	int n;
	Q(){n=0;}
	void clear()
	{
		for (int i=0;i<n;i++) delete a[i];
		if (n) delete a;n=0;
	}
	Q(int nn)//不能传入不是 2 的幂的数！
	{
		n=nn;
		assert(n==(n&-n));
		a=new ui*[n];
		for (int i=0;i<n;i++) a[i]=new ui[n],memset(a[i],0,n*sizeof a[0][0]);
	}
	const Q & operator=(const Q& b)
	{
		clear();n=b.n;
		a=new ui*[n];
		for (int i=0;i<n;i++) a[i]=new ui[n],memcpy(a[i],b.a[i],n*sizeof a[0][0]);
		return *this;
	}
	~Q(){clear();}
	Q operator+(const Q &b)
	{
		Q c(n);
		for (int i=0;i<n;i++) for (int j=0;j<n;j++) if ((c.a[i][j]=a[i][j]+b.a[i][j])>=p) c.a[i][j]-=p;
		return c;
	}
	Q operator-(const Q &b)
	{	
		Q c(n);
		for (int i=0;i<n;i++) for (int j=0;j<n;j++) if ((c.a[i][j]=a[i][j]-b.a[i][j])&fh) c.a[i][j]+=p;
		return c;
	}
	Q operator*(Q &b)
	{
		Q c(n);
		if (n<=128)
		{
			for (int i=0;i<n;i++) for (int k=0;k<n;k++) for (int j=0;j<n;j++) c.a[i][j]=(c.a[i][j]+(ull)a[i][k]*b.a[k][j])%p;
			return c;
		}
		Q A[2][2],B[2][2],s[10],p[5];
		n>>=1;
		int i,j,k,l;
		for (i=0;i<2;i++) for (j=0;j<2;j++)
		{
			A[i][j]=Q(n);
			for (k=0;k<n;k++) memcpy(A[i][j].a[k],a[k+i*n]+j*n,n*sizeof a[0][0]);
			B[i][j]=Q(n);
			for (k=0;k<n;k++) memcpy(B[i][j].a[k],b.a[k+i*n]+j*n,n*sizeof a[0][0]);
		}
		s[0]=B[0][1]-B[1][1];
		s[1]=A[0][0]+A[0][1];
		s[2]=A[1][0]+A[1][1];
		s[3]=B[1][0]-B[0][0];
		s[4]=A[0][0]+A[1][1];
		s[5]=B[0][0]+B[1][1];
		s[6]=A[0][1]-A[1][1];
		s[7]=B[1][0]+B[1][1];
		s[8]=A[0][0]-A[1][0];
		s[9]=B[0][0]+B[0][1];
		p[0]=A[0][0]*s[0];
		p[1]=s[1]*B[1][1];
		p[2]=s[2]*B[0][0];
		p[3]=A[1][1]*s[3];
		p[4]=s[4]*s[5];
		A[0][0]=p[4]+p[3]-p[1]+s[6]*s[7];
		A[0][1]=p[0]+p[1];
		A[1][0]=p[2]+p[3];
		A[1][1]=p[4]+p[0]-p[2]-s[8]*s[9];
		for (i=0;i<2;i++) for (j=0;j<2;j++)	for (k=0;k<n;k++) memcpy(c.a[k+i*n]+j*n,A[i][j].a[k],n*sizeof a[0][0]);
		n<<=1;
		return c;
	}
};
int main()
{
	int i,j,n,m,k;
	ios::sync_with_stdio(0);cin.tie(0);
	cin>>n>>m>>k;
	int N=1<<32-min({__builtin_clz(n-1),__builtin_clz(m-1),__builtin_clz(k-1)});
	Q a(N),b(N);
	for (i=0;i<n;i++) for (j=0;j<m;j++) cin>>a.a[i][j];
	for (i=0;i<m;i++) for (j=0;j<k;j++) cin>>b.a[i][j];
	a=a*b;
	for (i=0;i<n;i++) for (j=0;j<k;j++) cout<<a.a[i][j]<<" \n"[j+1==k]; 
}
```



#### exgcd

$O(\log p)$，$O(\log p)$。

```cpp
int exgcd(int a,int b,int c)//ax+by=c
{
	if (a==0) return c/b;
	return (c-(ll)b*exgcd(b%a,a,c))/a%b;
}
```

#### exBSGS

$O(\sqrt n)$。

```cpp
namespace BSGS
{
	typedef unsigned int ui;
	typedef unsigned long long ll;
	template<int N,typename T,typename TT> struct ht//个数，定义域，值域
	{
		const static int p=1e6+7,M=p+2;
		TT a[N];
		T v[N];
		int fir[p+2],nxt[N],st[p+2];//和模数相适应
		int tp,ds;//自定义模数
		ht(){memset(fir,0,sizeof fir);tp=ds=0;}
		void mdf(T x,TT z)//位置，值
		{
			ui y=x%p;
			for (int i=fir[y];i;i=nxt[i]) if (v[i]==x) return a[i]=z,void();//若不可能重复不需要 for
			v[++ds]=x;a[ds]=z;
			if (!fir[y]) st[++tp]=y;
			nxt[ds]=fir[y];fir[y]=ds;
		}
		TT find(T x)
		{
			ui y=x%p;
			int i;
			for (i=fir[y];i;i=nxt[i]) if (v[i]==x) return a[i];
			return 0;//返回值和是否判断依据要求决定
		}
		void clear()
		{
			++tp;
			while (--tp) fir[st[tp]]=0;
			ds=0;
		}
	};
	const int N=5e4;
	ht<N,ui,ui> s;
	int exgcd(int a,int b)
	{
		if (a==1) return 1;
		return (1-(long long)b*exgcd(b%a,a))/a;//not ll
	}
	int bsgs(ui a,ui b,ui p)
	{
		s.clear();
		a%=p;b%=p;
		if (!a) return 1-min((int)b,2);//含 -1
		ui i,j,k,x,y;
		x=sqrt(p)+2;
		for (i=0,j=1;i<x;i++,j=(ll)j*a%p)
		{
			if (j==b) return i;
			s.mdf((ll)j*b%p,i+1);
		}
		k=j;
		for (i=1;i<=x;i++,j=(ll)j*k%p) if (y=s.find(j)) return (ll)i*x-y+1;
		return -1;
	}
	bool isprime(ui p)
	{
		if (p<=1) return 0;
		for (ui i=2;i*i<=p;i++) if (p%i==0) return 0;
		return 1;
	}
	int exbsgs(ui a,ui b,ui p)//a^x=b(mod p)
	{
		//if (isprime(p)) return bsgs(a,b,p);
		a%=p;b%=p;
		ui i,j,k,x,y=__lg(p),cnt=0;
		for (i=0,j=1%p;i<=y;i++,j=(ll)j*a%p) if (j==b) return i;
		y=1;
		while (1)
		{
			if ((x=gcd(a,p))==1) break;
			if (b%x) return -1;//no sol
			++cnt;
			p/=x;b/=x;
			y=(ll)y*(a/x)%p;
		}
		a%=p;
		b=(ll)b*(p+exgcd(y,p))%p;
		int r=bsgs(a,b,p);
		return r==-1?-1:r+cnt;
	}
}
using BSGS::bsgs,BSGS::exbsgs;
```

#### exLucas

```cpp
namespace exlucas
{
	typedef long long ll;
	typedef pair<int,int> pa;
	int P,p,q,i;
	vector<pa> a;
	vector<vector<int> > b;
	vector<int> ph;
	vector<int> xs;
	int ksm(unsigned int x,ll y,const unsigned int p)
	{
		unsigned int r=1;
		while (y)
		{
			if (y&1) r=(unsigned long long)r*x%p;
			x=(unsigned long long)x*x%p;
			y>>=1;
		}
		return r;
	}
	void init(int x)//分解质因数，如有必要可以使用更快的方法
	{
		a.clear();b.clear();
		int i,y,z;
		vector<int> v;
		for (i=2;i*i<=x;i++) if (x%i==0)
		{
			z=i;x/=i;
			while (1)
			{
				y=x/i;
				if (i*y==x) x=y; else break;
				z*=i;
			}
			a.push_back(pa(i,z));
			b.push_back(v);
		}
		if (x>1) a.push_back(pa(x,x)),b.push_back(v);
		ph.resize(a.size());
		xs.resize(a.size());
		for (int k=0;k<a.size();k++)
		{
			tie(q,p)=a[k];
			ph[k]=p/q*(q-1);
			xs[k]=(ll)ksm(P/p,ph[k]-1,p)*(P/p)%P;
		}
	}
	void spinit(int x)//O(p) space
	{
		for (int k=0;k<a.size();k++)
		{
			int q,p;
			tie(q,p)=a[k];
			b[k].resize(p);
			b[k][0]=1;
			for (int i=1,j=q;i<p;i++) if (i==j) j+=q,b[k][i]=b[k][i-1]; else b[k][i]=(ll)b[k][i-1]*i%p;
		}
	}
	ll g(ll n)
	{
		ll r=0,s;
		while (n>=q)
		{
			n/=q;
			r+=n;
		}
		return r;
	}
	// int f(ll n)
	// {
	// 	if (n==0) return 1;
	// 	int r=1;//若 p>1e9 j 要 unsigned
	// 	for (int i=1,j=q;i<p;i++) if (i==j) j+=q; else r=(ll)r*i%p;
	// 	r=(ll)ksm(r,n/p,p)*f(n/q)%p;
	// 	n%=p;
	// 	for (int i=1,j=q;i<=n;i++) if (i==j) j+=q; else r=(ll)r*i%p;
	// 	return r;
	// }//O(T\sum p) time,O(1) space ver.
	int f(ll n)
	{
		int r=1;
		ll cs=0;
		while (n)
		{
			r=(ll)r*b[i][n%p]%p;
			cs+=n/p;
			n/=q;
		}
		return (ll)ksm(b[i][p-1],cs%ph[i],p)*r%p;
	}//O(\sum p) time,O(p) space ver.
	int C(ll n,ll m,int M)
	{
		if (n<m) return 0;
		int r=0,w;
		if (P!=M) init(P=M),spinit(P);//sp for O(p) space
		for (i=0;i<a.size();i++)
		{
			tie(q,p)=a[i];
			w=(ll)ksm(q,g(n)-g(m)-g(n-m),p)*f(n)%p*ksm((ll)f(m)*f(n-m)%p,ph[i]-1,p)%p;
			r=(r+(ll)xs[i]*w)%M;
		}
		return r;
	}
}
#define C(x,y,z) exlucas::C(x,y,z)
```

#### 杜教筛
```cpp
namespace du_seive
{
	typedef unsigned int ui;
	typedef unsigned long long ll;
	unordered_map<ll,ui> mp;
	const int N=1e7+2;
	const ui p=998244353;
	ui pr[N],phi[N];
	ui cnt;
	void init()
	{
		cnt=0;phi[1]=1;
		int i,j;
		for (i=2;i<N;i++)
		{
			if (!phi[i])
			{
				pr[cnt++]=i;
				phi[i]=i-1;
			}
			for (j=0;i*pr[j]<N;j++)
			{
				if (i%pr[j]==0)
				{
					phi[i*pr[j]]=phi[i]*pr[j];
					break;
				}
				phi[i*pr[j]]=phi[i]*(pr[j]-1);
			}
			if ((phi[i]+=phi[i-1])>=p) phi[i]-=p;
		}
	}
	ui get_phi_sum(ll n)
	{
		if (n<N) return phi[n];
		if (mp.count(n)) return mp[n];
		ui sum=0;
		for (ll i=2,j,k;i<=n;i=j+1)
		{
			j=n/(k=n/i);
			sum=(sum+(ll)get_phi_sum(k)*(j-i+1))%p;
		}
		ui nn=n%p;
		sum=(nn*(nn+1ll)/2+p-sum)%p;
		mp[n]=sum;
		return sum;
	}
}
using du_seive::init,du_seive::get_phi_sum;
```

#### 斐波那契数列

```cpp
const int NN=3e7+2,M=4e5,N=1e6+10;
char c[NN];
ll n;
ll y,mo,x,z;
int p,i,j,k;
struct Q
{
	int a[2][2];
	Q(int b=0,int c=0,int d=0,int e=0){a[0][0]=b,a[0][1]=c,a[1][0]=d,a[1][1]=e;}
	Q operator*(const Q &o)
	{
		return Q(((ll)a[0][0]*o.a[0][0]+(ll)a[0][1]*o.a[1][0])%p,
				((ll)a[0][0]*o.a[0][1]+(ll)a[0][1]*o.a[1][1])%p,
				((ll)a[1][0]*o.a[0][0]+(ll)a[1][1]*o.a[1][0])%p,
				((ll)a[1][0]*o.a[0][1]+(ll)a[1][1]*o.a[1][1])%p);
	}
};
struct ht
{
	ll v[N],a[N];
	int fir[N],nxt[N],st[N];//和模数相适应
	int tp,p,ds;//自定义模数
	ht(){tp=0,p=1e6+7,ds=0;}
	void mdf(const ll x,const ll z)//位置，值
	{
		const int y=x%p;
		for (int i=fir[y];i;i=nxt[i]) if (v[i]==x) return a[i]=z,void();//若不可能重复不需要这一步if，但需要for?
		v[++ds]=x;a[ds]=z;if (!fir[y]) st[++tp]=y;
		nxt[ds]=fir[y];fir[y]=ds;
	}
	ll find(const ll x)
	{
		const int y=x%p;int i;
		for (i=fir[y];i;i=nxt[i]) if (v[i]==x) break;
		if (!i) return 0;//返回值和是否判断依据要求决定
		return a[i];
	}
	void clear()
	{
		++tp;
		while (--tp) fir[st[tp]]=0;ds=0;
	}
};
ht mp;
Q f[M],g[M],ji;
int fib(ll n)
{
	Q x=f[n%k]*g[n/k];
	return x.a[0][1];
}
ll spefib(ll n)
{
	Q x=f[n%k]*g[n/k];
	return (ll)x.a[0][1]*p+x.a[1][1];
}
ll sj()
{
	ll x=rand();
	x=x<<15^rand();
	x=x<<15^rand();
	x=x<<15^rand();
	return x>0?x:-x;
}
ll ab(ll x)
{
	return x>0?x:-x;
}
int main()
{
	srand(383778817);
	scanf("%s\n%d",c+1,&p);
	k=sqrt((ll)20*p)+1;ji=Q(0,1,1,1);
	f[0]=Q(1,0,0,1);for (i=1;i<=k;i++) f[i]=f[i-1]*ji;
	g[0]=Q(1,0,0,1);for (i=1;i<=k;i++) g[i]=g[i-1]*f[k];
	while (1)
	{
		x=sj()%(20ll*p)+1;y=spefib(x);
		if (z=mp.find(y))
		{
			if (z!=x)
			{
				mo=ab(x-z);
				break;
			}
		} else mp.mdf(y,x);
	}
	n=0;
	for (i=1;c[i]>=48&&c[i]<=57;i++) n=(n*10+(c[i]^48))%mo;
	printf("%d",fib(n));
}
```

#### 线性插值（$k$ 次幂和）

$O(m)$，$O(m)$。

```cpp
int f(int *a,int n,int m)//这种写法不包含0处取值，n是值，m-1是次数
{
	if (n<=m) return a[n];
	static int inv[N],l[N],r[N],ifac[N]; 
	int i;
	ifac[0]=inv[1]=1;
	for (i=2;i<=m;i++) inv[i]=p-(ll)p/i*inv[p%i]%p;
	for (i=1;i<=m;i++) ifac[i]=(ll)ifac[i-1]*inv[i]%p;//以上可以预跑
	int ans=0,rr=0;
	l[0]=1;r[m+1]=1;
	for (i=1;i<m;i++) l[i]=(ll)l[i-1]*(n-i)%p;
	for (i=m;i;i--) r[i]=(ll)r[i+1]*(n-i)%p;
	for (i=1;i<=m;i++)
	{
		if ((m^i)&1) rr=p-a[i]; else rr=a[i];
		ans=(ans+(ll)rr*ifac[i-1]%p*ifac[m-i]%p*l[i-1]%p*r[i+1])%p;
	}
	return ans;
}
```

#### 单原根（仅手动验证质数）
```cpp
namespace get_root
{
	typedef unsigned int ui;
	typedef unsigned long long ll;
	ui ksm(ui x,ui y,ui p)
	{
		ui r=1;
		while (y)
		{
			if (y&1) r=(ll)r*x%p;
			x=(ll)x*x%p;y>>=1;
		}
		return r;
	}
	vector<ui> getw(ui n)
	{
		vector<ui> w;
		for (ui i=2;i*i<=n;i++) if (n%i==0)
		{
			w.push_back(i);
			n/=i;
			for (ui j=n/i;n==i*j;j=n/i) n/=i;
		}
		if (n>1) w.push_back(n);
		return w;
	}
	int getrt(ui n)
	{
		if (n<=2) return n-1;
		auto w=getw(n);
		ui ph=n;
		for (ui x:w) ph=ph/x*(x-1);
		w=getw(ph);
		for (ui &x:w) x=ph/x;
		for (ui i=2;i<n;i++) if (gcd(i,n)==1)
		{
			for (ui x:w) if (ksm(i,x,n)==1) goto no;
			return i;
			no:;
		}
		return -1;
	}
}
using get_root::getrt;
```

#### 筛全部原根

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=1e6+2;
int ss[N],mn[N],fmn[N],phi[N];
int t,n,gs,i,d;
bool ed[N],av[N],yg[N],hv[N];
double inv[N];
void getfac(int x,int *a,int &n)
{
	int y=x,z;
	if (1^x&1)
	{
		a[n=1]=2;x>>=1;while (1^x&1) x>>=1;
	}
	while (x>1)
	{
		x=1e-9+(x*inv[a[++n]=z=mn[x]]);
		while (x%z==0) x=1e-9+x*inv[z];
	}
	for (i=1;i<=n;i++) av[a[i]]=0,a[i]=1e-9+(y*inv[a[i]]);
}
int ksm(int x,int y,int p)
{
	int r=1;
	while (y)
	{
		if (y&1) r=(ll)r*x%p;
		x=(ll)x*x%p;y>>=1;
	}
	return r;
}
bool ck(int x,int *a,int n,int p)
{
	for (int i=1;i<=n;i++) if (ksm(x,a[i],p)==1) return 0;
	return 1;
} 
void getrt(int x,int d)
{
	if (!hv[x]) return puts("0\n"),void();
	static int a[30];
	int n=0,y,i,g=0,c=d;y=phi[x];
	fill(av+1,av+y+1,1);
	getfac(y,a,n);
	for (i=1;i<x;i++) if (__gcd(i,x)==1&&ck(i,a,n,x)) break;
	yg[g=i]=1;//g就是最小原根
	int j=(ll)g*g%x;
	for (i=2;i<y;i++,j=(ll)j*g%x) yg[j]=av[i]=av[mn[i]]&av[fmn[i]]; 
	printf("%d\n",phi[y]);
	for (i=1;i<x;i++) if (yg[i]) 
	{
		yg[i]=0;
		if (--c==0) printf("%d ",i),c=d;
	}puts("");
}
void init()
{
	int i,j,k,n=N-1;
	mn[1]=phi[1]=1;
	for (i=1;i<=n;i++) inv[i]=1.0/i;
	for (i=2;i<=n;i++)
	{
		if (!ed[i]) phi[mn[i]=ss[++gs]=i]=i-1,hv[i]=1;
		for (j=1;j<=gs&&(k=ss[j]*i)<=n;j++)
		{
			ed[k]=1;mn[k]=ss[j];
			if (i%ss[j]==0) {phi[k]=phi[i]*ss[j];hv[k]=hv[i];break;}
			phi[k]=phi[i]*(ss[j]-1);
		}
	}
	for (i=n;i;i--) fmn[i]=1e-9+(i*inv[mn[i]]),hv[i]|=(1^i&1)&&hv[i>>1];
	for (i=8;i<=n;i<<=1) hv[i]=0;
}
int main()
{
	init();
	scanf("%d",&t);
	while (t--)
	{
		scanf("%d%d",&n,&d);
		getrt(n,d);
	}
}
```



#### 圆上整点

```cpp
	while (n&1^1) n>>=1;//n是半径
	for (i=3;i<=sqrt(n);i++) if (n%i==0)
	{
		if (i%4==3)
		{
			while (n%i==0) n/=i;
		}
		else
		{
			k=0;
			while (n%i==0)
			{
				n/=i;++k;
			}
			ans*=k<<1|1;
		}
	}
	if ((n>1)&&(n%4==1)) ans*=3;
	printf("%d",ans<<2);
```

#### 行列式求值（任意模数）

$O(n^3)$，$O(n^2)$。

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=502,p=998244353;
int cal(int a[][N],int n)
{
	int i,j,k,r=1,fh=0,l;
	for (i=1;i<=n;i++)
	{
		k=i;
		for (j=i+1;j<=n;j++) if (a[j][i]) {k=j;break;}
		if (a[k][i]==0) return 0;
		if (i!=k) {swap(a[k],a[i]);fh^=1;}
		for (j=i+1;j<=n;j++)
		{
			if (a[j][i]>a[i][i]) swap(a[j],a[i]),fh^=1;
			while (a[j][i])
			{
				l=a[i][i]/a[j][i];
				for (k=i;k<=n;k++) a[i][k]=(a[i][k]+(ll)(p-l)*a[j][k])%p;
				swap(a[j],a[i]);fh^=1;
			}
		}
		r=(ll)r*a[i][i]%p;
	}
	if (fh) return (p-r)%p;
	return r;
}
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	int n,i,j;
	static int a[N][N];
	cin>>n;
	for (i=1;i<=n;i++) for (j=1;j<=n;j++) cin>>a[i][j];
	cout<<cal(a,n)<<endl;
}
```

#### 行列式求值（质数模数）

$O(n^3)$，$O(n^2)$。

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=502,p=998244353;
int ksm(int x,int y)
{
	int r=1;
	while (y)
	{
		if (y&1) r=(ll)r*x%p;
		y>>=1;x=(ll)x*x%p;
	}
	return r;
}
int cal(int a[][N],int n)
{
	int i,j,k,r=1,fh=0,l;
	for (i=1;i<=n;i++)
	{
		for (j=i;j<=n;j++) if (a[j][i]) break;
		if (j>n) return 0;
		if (i!=j) swap(a[j],a[i]),fh^=1;
		r=(ll)r*a[i][i]%p;
		k=ksm(a[i][i],p-2);
		for (j=i;j<=n;j++) a[i][j]=(ll)a[i][j]*k%p;
		for (j=i+1;j<=n;j++)
		{
			a[j][i]=p-a[j][i];
			for (k=i+1;k<=n;k++) a[j][k]=(a[j][k]+(ll)a[j][i]*a[i][k])%p;
			a[j][i]=0;
		}
	}
	if (fh) return (p-r)%p;
	return r;
}
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	int n,i,j;
	static int a[N][N];
	cin>>n;
	for (i=1;i<=n;i++) for (j=1;j<=n;j++) cin>>a[i][j];
	cout<<cal(a,n)<<endl;
}
/*
3
3 1 4
1 5 9
2 6 5
998244263
*/
```

#### Min_25 筛（$f(p^k)=p^k(p^k-1)$，求 $\sum\limits_{i=1}^nf(i)$）

```cpp
const int N=1e5+2,p=1e9+7,i6=166666668;
ll fs[N<<1],m;
int ss[N],ys[N<<1],s[N],f[N<<1],g[N<<1],ls[N<<1],cs[N<<1];
int gs,n,i,j,k,cnt,ct,ans,sq;
bool ed[N];
int S(ll n,int x)
{
	int r,i,j,l;
	ll k;
	if (ss[x]>=n) return 0;
	if (n>sq) r=g[ys[m/n]]; else r=g[n];
	if ((r=r-s[x])<0) r+=p;
	for (i=x+1;(ll)ss[i]*ss[i]<=n;i++) for (j=1,k=ss[i];k<=n;j++,k*=ss[i])
	{
		l=(k-1)%p;
		r=(r+(ll)l*(l+1)%p*((j!=1)+S(n/k,i)))%p;
	}
	return r;
}
int main()
{
	n=1e5;
	for (i=2;i<=n;i++)
	{
		if (!ed[i]) ss[++gs]=i;
		for (j=1;(j<=gs)&&(i*ss[j]<=n);j++)
		{
			ed[i*ss[j]]=1;
			if (i%ss[j]==0) break;
		}
	}ss[gs+1]=1e6;
	s[1]=ss[1]*ss[1];
	for (i=2;i<=gs;i++) s[i]=(s[i-1]+(ll)ss[i]*ss[i])%p;//s 是多项式在素数位置的前缀和
	memcpy(cs,s,sizeof(s));
	ll i,j,k,x,z; scanf("%lld",&m);
	sq=n=sqrt(m);while ((ll)(n+1)*(n+1)<=m) ++n;
	cnt=n-1;
	for (i=n;i<=m;i=j+1) {j=m/(m/i);++cnt;}ct=cnt++;
	for (i=1;i<=m;i=j+1)
	{
		j=m/(k=m/i);
		if (k<=n) g[fs[k]=k]=(k*(k+1)*(k<<1|1)/6-1)%p;//这里是多项式前缀和（不含1）
		else
		{
			z=k%p;//一样
			g[ys[j]=--cnt]=(z*(z+1)%p*(z<<1|1)%p+p-6)*i6%p;fs[cnt]=k;
		}
	}
	cnt=ct;
	for (j=1;(j<=gs)&&(z=(ll)ss[j]*ss[j]);j++) for (i=cnt;z<=fs[i];i--)
	{
		x=fs[i]/ss[j];if (x>n) x=ys[m/x];
		g[i]=(g[i]+(ll)(p-ss[j])*ss[j]%p*(g[x]-s[j-1]+p))%p;//另一处需要修改的
	}
	memcpy(ls,g,sizeof(g));
	s[1]=ss[1];
	for (i=2;i<=gs;i++) s[i]=s[i-1]+ss[i];
	cnt=n-1;
	for (i=n;i<=m;i=j+1) {j=m/(m/i);++cnt;}ct=cnt++;
	for (i=1;i<=m;i=j+1)
	{
		j=m/(k=m/i);
		if (k<=n) g[fs[k]=k]=((k*(k+1)>>1)-1)%p;
		else
		{
			z=k%p;
			g[ys[j]=--cnt]=(z*(z+1)-2>>1)%p;fs[cnt]=k;
		}
	}
	cnt=ct;
	for (j=1;(j<=gs)&&(z=(ll)ss[j]*ss[j]);j++) for (i=cnt;z<=fs[i];i--)
	{
		x=fs[i]/ss[j];if (x>n) x=ys[m/x];
		g[i]=(g[i]+(ll)(p-ss[j])*(g[x]-s[j-1]+p))%p;
	}
	for (i=1;i<=cnt;i++) if ((g[i]=ls[i]-g[i])<0) g[i]+=p;
	for (i=1;i<=gs;i++) if ((s[i]=cs[i]-s[i])<0) s[i]+=p;
	ans=S(m,0)+1;if (ans==p) ans=0;printf("%d",ans);
}
```

#### Min_25筛（卡常，素数个数，注意评测机 double 性能）

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=3.2e5+2;
ll s[N];
int ss[N],ys[N],gs=0;
bool ed[N];
ll cal(ll m)
{
	static ll g[N<<1],fs[N<<1];
	ll i,j,k,x;
	int n;
	int p,q,cnt;
	n=round(sqrt(m));
	q=lower_bound(ss+1,ss+gs+1,n)-ss;
	memset(g,0,sizeof(g));memset(ys,0,sizeof(ys));cnt=n-1;
	for (i=n;i<=m;i=j+1) {j=m/(m/i);++cnt;}int ct=cnt++;
	for (i=1;i<=m;i=j+1)
	{
		j=m/(k=m/i);
		if (k<=n) g[fs[k]=k]=k-1; else {g[ys[j]=--cnt]=k-1;fs[cnt]=k;}
	}cnt=ct;
	for (j=1;j<=q;j++) for (i=cnt;(ll)ss[j]*ss[j]<=fs[i];i--)
	{
		x=fs[i]/ss[j];if (x>n) x=ys[m/x];
		g[i]-=g[x]-j+1;
	}
	return g[cnt];//这里 g[cnt-i+1] 表示的是 [1,m/i] 的答案
}
int main()
{
	int n,i,j,t;
	n=3.2e5;
	for (i=2;i<=n;i++)
	{
		if (!ed[i]) ss[++gs]=i;
		for (j=1;(j<=gs)&&(i*ss[j]<=n);j++)
		{
			ed[i*ss[j]]=1;
			if (i%ss[j]==0) break;
		}
	}
	s[1]=ss[1];
	for (i=2;i<=gs;i++) s[i]=s[i-1]+ss[i];
	t=1;
	ll m;
	while (t--) cin>>m,cout<<cal(m)<<'\n';
}
```

#### 扩展 min-max 容斥（重返现世）

$\max\limits_{k}\{S\}=\sum\limits_{T\subseteq S}(-1)^{|T|-k}\tbinom{|T|-1}{k-1}\min\{T\}$

```cpp
	scanf("%d%d%d",&n,&q,&m);inv[1]=1;q=n+1-q;
	for (i=2;i<=m;i++) inv[i]=p-(ll)p/i*inv[p%i]%p;
	for (i=1;i<=n;i++) scanf("%d",a+i);f[0][0]=1;
	for (j=1;j<=n;j++) for (i=q;i;i--) for (k=m;k>=a[j];k--) if ((f[i][k]=f[i][k]+f[i-1][k-a[j]]-f[i][k-a[j]])>=p) f[i][k]-=p; else if (f[i][k]<0) f[i][k]+=p;
	for (i=1;i<=m;i++) ans=(ans+(ll)f[q][i]*inv[i])%p;
	ans=(ll)ans*m%p;printf("%d",ans);
```

#### 模数为偶数 FWT & 光速乘

$O(n2^n)$，$O(2^n)$。

```cpp
const int N=1<<20,M=21;
int x[M];
ll p,f[N],g[N];
int n,m,c;
ll mul(ll x,ll y)
{
	x=x*y-(ll)((ldb)x/p*y+1e-8)*p;
	if (x<0) return x+p;return x;
}
void read(int &x)
{
	c=getchar();
	while ((c<48)||(c>57)) c=getchar();
	x=c^48;c=getchar();
	while ((c>=48)&&(c<=57))
	{
		x=x*10+(c^48);
		c=getchar();
	}
}
void read(ll &x)
{
	c=getchar();
	while ((c<48)||(c>57)) c=getchar();
	x=c^48;c=getchar();
	while ((c>=48)&&(c<=57))
	{
		x=x*10+(c^48);
		c=getchar();
	}
}
void dft(ll *a)
{
	int i,j,k,l;
	ll b;
	for (i=1;i<n;i=l)
	{
		l=i<<1;
		for (j=0;j<n;j+=l) for (k=0;k<i;k++)
		{
			b=a[j|k|i];
			a[j|k|i]=(a[j|k]-b+p)%p;
			a[j|k]=(a[j|k]+b)%p;
		}
	}
}
int main()
{
	ll t;int i;
	read(m);read(t);read(p);p*=(n=1<<m);
	for (i=0;i<n;i++) read(f[i]);
	dft(f);
	for (i=0;i<=m;i++) read(x[i]);
	for (i=1;i<n;i++) g[i]=g[i>>1]+(i&1);
	for (i=0;i<n;i++) g[i]=x[g[i]];dft(g);
	while (t)
	{
		if (t&1) for (i=0;i<n;i++) f[i]=mul(f[i],g[i]);
		for (i=0;i<n;i++) g[i]=mul(g[i],g[i]);t>>=1;
	}
	dft(f);
	for (i=0;i<n;i++) printf("%d\n",int(f[i]>>m));
}
```

#### FWT

$O(n2^n)$，$O(2^n)$。

```cpp
void fwt_and(vector<ui> &A)//本质：母集和
{
	ui n=A.size(),*a=A.data(),i,j,k,l,*f,*g;
	for (i=1;i<n;i=l)
	{
		l=i*2;
		for (j=0;j<n;j+=l)
		{
			f=a+j;g=a+j+i;
			for (k=0;k<i;k++) if ((f[k]+=g[k])>=p) f[k]-=p;
		}
	}
}
void ifwt_and(vector<ui> &A)
{
	ui n=A.size(),*a=A.data(),i,j,k,l,*f,*g;
	for (i=1;i<n;i=l)
	{
		l=i*2;
		for (j=0;j<n;j+=l)
		{
			f=a+j;g=a+j+i;
			for (k=0;k<i;k++) if ((f[k]-=g[k])>=p) f[k]+=p;//unsigned
		}
	}
}
void fwt_or(vector<ui> &A)//本质：子集和
{
	ui n=A.size(),*a=A.data(),i,j,k,l,*f,*g;
	for (i=1;i<n;i=l)
	{
		l=i*2;
		for (j=0;j<n;j+=l)
		{
			f=a+j;g=a+j+i;
			for (k=0;k<i;k++) if ((g[k]+=f[k])>=p) g[k]-=p;
		}
	}
}
void ifwt_or(vector<ui> &A)
{
	ui n=A.size(),*a=A.data(),i,j,k,l,*f,*g;
	for (i=1;i<n;i=l)
	{
		l=i*2;
		for (j=0;j<n;j+=l)
		{
			f=a+j;g=a+j+i;
			for (k=0;k<i;k++) if ((g[k]-=f[k])>=p) g[k]+=p;//unsigned
		}
	}
}
void fwt_xor(vector<ui> &A)
{
	ui n=A.size(),*a=A.data(),i,j,k,l,*f,*g;
	for (i=1;i<n;i=l)
	{
		l=i*2;
		for (j=0;j<n;j+=l)
		{
			f=a+j;g=a+j+i;
			for (k=0;k<i;k++)
			{
				if ((f[k]+=g[k])>=p) f[k]-=p;
				g[k]=(f[k]+2*(p-g[k]))%p;
			}
		}
	}
}
void ifwt_xor(vector<ui> &A)
{
	ui n=A.size(),*a=A.data(),i,j,k,l,*f,*g,x=p+1>>1,y=1;
	for (i=1;i<n;i=l)
	{
		l=i*2;
		for (j=0;j<n;j+=l)
		{
			f=a+j;g=a+j+i;
			for (k=0;k<i;k++)
			{
				if ((f[k]+=g[k])>=p) f[k]-=p;
				g[k]=(f[k]+2*(p-g[k]))%p;
			}
		}
		y=(ll)y*x%p;
	}
	for (i=0;i<n;i++) a[i]=(ll)a[i]*y%p;
}
```

#### 二次剩余
```cpp
namespace cipolla
{
	typedef unsigned int ui;
	typedef unsigned long long ll;
	ui p,w;
	struct Q
	{
		ll x,y;
		Q operator*(const Q &o) const {return {(x*o.x+y*o.y%p*w)%p,(x*o.y+y*o.x)%p};}
	};
	ui ksm(ll x,int y)
	{
		ll r=1;
		while (y)
		{
			if (y&1) r=r*x%p;
			x=x*x%p;y>>=1;
		}
		return r;
	}
	Q ksm(Q x,int y)
	{
		Q r={1,0};
		while (y)
		{
			if (y&1) r=r*x;
			x=x*x;y>>=1;
		}
		return r;
	}
	int mosqrt(ui x,ui P)//0<=x<P
	{
		if (x==0||P==2) return x;
		p=P;
		if (ksm(x,p-1>>1)!=1) return -1;
		ui y;
		mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
		do y=rnd()%p,w=((ll)y*y+p-x)%p; while (ksm(w,p-1>>1)<=1);//not for p=2
		y=ksm({y,1},p+1>>1).x;
		if (y*2>p) y=p-y;//两解取小
		return y;
	}
}
using cipolla::mosqrt;
```

#### NTT

```cpp
#include <bits/stdc++.h>
using namespace std;
#if !defined(ONLINE_JUDGE)&&defined(LOCAL)
#include "my_header\debug.h"
#else
#define dbg(...); 1;
#endif
std::mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
namespace NTT//禁止混用三模与普通 NTT
{
	#define all(x) (x).begin(),(x).end()
	const int N=1<<21;//务必修改
	//#define MTT
	//#define normal
	typedef unsigned int ui;
	typedef unsigned long long ll;
	const ui g=3,f=1u<<31,I=86'583'718;//g^((p-1)/4)
#ifndef MTT
	const ui p=998244353;
	ui w[N];
#else
	const ui p=1e9+7;
	const ui p1=469'762'049,p2=998'244'353,p3=1'004'535'809;//三模，原根都是 3，非常好
	const ui inv_p1=554'580'198,inv_p12=395'249'030;//三模，1 关于 2 逆，1*2 关于 3 逆，1*2 mod 3
	ui w1[N],w2[N],w3[N];//三模
#endif
	ui r[N],preone=0;
	ui inv[N],fac[N],ifac[N],preinv=0,prefac=-1,W;//W for mosqrt
	ui ksm(ui x,ui y)
	{
		ui r=1;
		while (y)
		{
			if (y&1) r=(ll)r*x%p;
			x=(ll)x*x%p;
			y>>=1;
		}
		return r;
	}
#ifdef MTT
	ui ksm1(ui x,ui y)
	{
		ui r=1;
		while (y)
		{
			if (y&1) r=(ll)r*x%p1;
			x=(ll)x*x%p1;
			y>>=1;
		}
		return r;
	}
	ui ksm2(ui x,ui y)
	{
		ui r=1;
		while (y)
		{
			if (y&1) r=(ll)r*x%p2;
			x=(ll)x*x%p2;
			y>>=1;
		}
		return r;
	}
	ui ksm3(ui x,ui y)
	{
		ui r=1;
		while (y)
		{
			if (y&1) r=(ll)r*x%p3;
			x=(ll)x*x%p3;
			y>>=1;
		}
		return r;
	}
#endif
	vector<ui> getinvs(vector<ui> a)
	{
		static ui l[N],r[N];
		int n=a.size(),i;
		if (n<=2)
		{
			for (i=0;i<n;i++) a[i]=ksm(a[i],p-2);
			return a;
		}
		l[0]=a[0];r[n-1]=a[n-1];
		for (i=1;i<n;i++) l[i]=(ll)l[i-1]*a[i]%p;
		for (i=n-2;i;i--) r[i]=(ll)r[i+1]*a[i]%p;
		ui x=ksm(l[n-1],p-2);
		a[0]=(ll)x*r[1]%p;a[n-1]=(ll)x*l[n-2]%p;
		for (i=1;i<n-1;i++) a[i]=(ll)x*l[i-1]%p*r[i+1]%p;
		return a;
	}
	struct P
	{
		ui x,y;
		P(ui a=0,ui b=0):x(a),y(b){}
		P operator*(P &a)
		{
			return P(((ll)x*a.x+(ll)y*a.y%p*W)%p,((ll)x*a.y+(ll)y*a.x)%p);
		}
	};
	ui ksm(P x,ui y)
	{
		P r(1,0);
		while (y)
		{
			if (y&1) r=r*x;
			x=x*x;y>>=1;
		}
		return r.x;
	}
	int mosqrt(ui x)
	{
		if (x==0) return 0;
		if (ksm(x,p-1>>1)!=1) {cerr<<"No mosqrt"<<endl;exit(0);}
		ui y;
		do y=rnd()%p; while (ksm(W=((ll)y*y%p+p-x)%p,p-1>>1)<=1);//not for p=2
		y=ksm(P(y,1),p+1>>1);
		return y*2<p?y:p-y;
	}
#ifdef MTT
	void init(ui n)
	{
		if (preone==n) return;
		ui b=__builtin_ctz(n)-1,i,j,k,l,wn;
		for (i=1;i<n;i++) r[i]=r[i>>1]>>1|(i&1)<<b;++b;
		if (preone<n)
		{
			for (j=1,k=1;j<n;j=l,k++)
			{
				l=j<<1;
				wn=ksm1(g,p1-1>>k);
				w1[j]=1;
				for (i=j+1;i<l;i++) w1[i]=(ll)w1[i-1]*wn%p1;
			}
			for (j=1,k=1;j<n;j=l,k++)
			{
				l=j<<1;
				wn=ksm2(g,p2-1>>k);
				w2[j]=1;
				for (i=j+1;i<l;i++) w2[i]=(ll)w2[i-1]*wn%p2;
			}
			for (j=1,k=1;j<n;j=l,k++)
			{
				l=j<<1;
				wn=ksm3(g,p3-1>>k);
				w3[j]=1;
				for (i=j+1;i<l;i++) w3[i]=(ll)w3[i-1]*wn%p3;
			}
		}
		preone=n;
	}
#else
	void init(ui n)
	{
		if (preone==n) return;
		ui b=__builtin_ctz(n)-1,i,j,k,l,wn;
		for (i=1;i<n;i++) r[i]=r[i>>1]>>1|(i&1)<<b;++b;
		if (preone<n)
		{
			for (j=1,k=1;j<n;j=l,k++)
			{
				l=j<<1;
				wn=ksm(g,p-1>>k);
				w[j]=1;
				for (i=j+1;i<l;i++) w[i]=(ll)w[i-1]*wn%p;
			}
		}
		preone=n;
	}
#endif
	ui cal(ui x) {return 1u<<32-__builtin_clz(max(x,2u)-1);}
	void getinv(int n)
	{
		if (!preinv) preinv=inv[1]=1;
		if (n<=preinv) return;
		for (ui i=preinv+1,j;i<=n;i++)
		{
			j=p/i;
			inv[i]=(ll)(p-j)*inv[p-i*j]%p;
		}
		preinv=n;
	}
	void getfac(int n)
	{
		if (prefac==-1) prefac=0,ifac[0]=fac[0]=1;
		if (n<=prefac) return;
		getinv(n);
		for (ui i=prefac+1,j;i<=n;i++) fac[i]=(ll)fac[i-1]*i%p,ifac[i]=(ll)ifac[i-1]*inv[i]%p;
		prefac=n;
	}
	struct Q
	{
		vector<ui> a;
		Q(ui x=2)
		{
			x=cal(x);
			a.resize(x);
		}
		ui fx(ui x)
		{
			ui r=0;
			int i;
			for (i=a.size()-1;i>=0&&!a[i];i--);
			for (;i>=0;i--) r=((ll)r*x+a[i])%p;
			return r;
		}
	#ifdef MTT
		void dft1(int o=0)
		{
			ui n=a.size(),i,j,k,x,y,*f,*g,*wn,*A=a.data();
			init(n);
			for (i=1;i<n;i++) if (i<r[i]) swap(A[i],A[r[i]]);
			for (k=1;k<n;k<<=1)
			{
				wn=w1+k;
				for (i=0;i<n;i+=k<<1) 
				{
					f=A+i;g=A+i+k;
					for (j=0;j<k;j++)
					{
						x=f[j];y=(ll)g[j]*wn[j]%p1;
						if (x+y>=p1) f[j]=x+y-p1; else f[j]=x+y;
						if (x<y) g[j]=x-y+p1; else g[j]=x-y;
					}
				}
			}
			if (o)
			{
				x=ksm1(n,p1-2);
				for (i=0;i<n;i++) A[i]=(ll)A[i]*x%p1;
				reverse(A+1,A+n);
			}
		}
		void dft2(int o=0)
		{
			ui n=a.size(),i,j,k,x,y,*f,*g,*wn,*A=a.data();
			init(n);
			for (i=1;i<n;i++) if (i<r[i]) swap(A[i],A[r[i]]);
			for (k=1;k<n;k<<=1)
			{
				wn=w2+k;
				for (i=0;i<n;i+=k<<1) 
				{
					f=A+i;g=A+i+k;
					for (j=0;j<k;j++)
					{
						x=f[j];y=(ll)g[j]*wn[j]%p2;
						if (x+y>=p2) f[j]=x+y-p2; else f[j]=x+y;
						if (x<y) g[j]=x-y+p2; else g[j]=x-y;
					}
				}
			}
			if (o)
			{
				x=ksm2(n,p2-2);
				for (i=0;i<n;i++) A[i]=(ll)A[i]*x%p2;
				reverse(A+1,A+n);
			}
		}
		void dft3(int o=0)
		{
			ui n=a.size(),i,j,k,x,y,*f,*g,*wn,*A=a.data();
			init(n);
			for (i=1;i<n;i++) if (i<r[i]) swap(A[i],A[r[i]]);
			for (k=1;k<n;k<<=1)
			{
				wn=w3+k;
				for (i=0;i<n;i+=k<<1) 
				{
					f=A+i;g=A+i+k;
					for (j=0;j<k;j++)
					{
						x=f[j];y=(ll)g[j]*wn[j]%p3;
						if (x+y>=p3) f[j]=x+y-p3; else f[j]=x+y;
						if (x<y) g[j]=x-y+p3; else g[j]=x-y;
					}
				}
			}
			if (o)
			{
				x=ksm3(n,p3-2);
				for (i=0;i<n;i++) A[i]=(ll)A[i]*x%p3;
				reverse(A+1,A+n);
			}
		}
	#else
		void dft(int o=0)
		{
			ui n=a.size(),i,j,k,x,y,*f,*g,*wn,*A=a.data();
			init(n);
			for (i=1;i<n;i++) if (i<r[i]) swap(A[i],A[r[i]]);
			for (k=1;k<n;k<<=1)
			{
				wn=w+k;
				for (i=0;i<n;i+=k<<1) 
				{
					f=A+i;g=A+i+k;
					for (j=0;j<k;j++)
					{
						x=f[j];y=(ll)g[j]*wn[j]%p;
						if (x+y>=p) f[j]=x+y-p; else f[j]=x+y;
						if (x<y) g[j]=x-y+p; else g[j]=x-y;
					}
				}
			}
			if (o)
			{
				getinv(n);x=inv[n];
				for (i=0;i<n;i++) A[i]=(ll)A[i]*x%p;
				reverse(A+1,A+n);
			}
		}
		void hf_dft(int o=0)
		{
			ui n=a.size()>>1,i,j,k,x,y,*f,*g,*wn,*A=a.data();
			init(n);
			for (i=1;i<n;i++) if (i<r[i]) swap(A[i],A[r[i]]);
			for (k=1;k<n;k<<=1)
			{
				wn=w+k;
				for (i=0;i<n;i+=k<<1) 
				{
					f=A+i;g=A+i+k;
					for (j=0;j<k;j++)
					{
						x=f[j];y=(ll)g[j]*wn[j]%p;
						if (x+y>=p) f[j]=x+y-p; else f[j]=x+y;
						if (x<y) g[j]=x-y+p; else g[j]=x-y;
					}
				}
			}
			if (o)
			{
				getinv(n);x=inv[n];
				for (i=0;i<n;i++) A[i]=(ll)A[i]*x%p;
				reverse(A+1,A+n);
			}
		}
	#endif
		Q dao()
		{
			ui n=a.size();
			Q r(n);
			for (ui i=1;i<n;i++) r.a[i-1]=(ll)a[i]*i%p;
			return r;
		}
		Q ji()
		{
			ui n=a.size();
			getinv(n-1);
			Q r(n);
			for (ui i=1;i<n;i++) r.a[i]=(ll)a[i-1]*inv[i]%p; 
			return r;
		}
		Q operator-() const {Q r=*this;ui n=a.size();for (ui i=0;i<n;i++) if (r.a[i]) r.a[i]=p-r.a[i];return r;}
		Q operator+(ui x) const {Q r=*this;r+=x;return r;}
		void operator+=(ui x) {if ((a[0]+=x)>=p) a[0]-=p;}
		Q operator-(ui x) const {Q r=*this;r-=x;return r;}
		void operator-=(ui x) {if (a[0]<x) a[0]=a[0]+p-x; else a[0]-=x;}
		Q operator*(ui k) const {Q r=*this;r*=k;return r;}
		void operator*=(ui k) {ui n=a.size();for (ui i=0;i<n;i++) a[i]=(ll)a[i]*k%p;}
		Q operator+(Q f) const {f+=*this;return f;}
		void operator+=(const Q &f) {ui n=f.a.size();if (a.size()<n) a.resize(n);for (ui i=0;i<n;i++) if ((a[i]+=f.a[i])>=p) a[i]-=p;}
		Q operator-(Q f) const {Q r=*this;r-=f;return r;}
		void operator-=(const Q &f) {ui n=f.a.size();if (a.size()<n) a.resize(n);for (ui i=0;i<n;i++) if (a[i]<f.a[i]) a[i]+=p-f.a[i]; else a[i]-=f.a[i];}
		Q operator*(Q f) const {f*=*this;return f;}
#ifdef normal
	#ifdef MTT
		void operator*=(Q g3)
		{
			cout<<"I didn't check it"<<endl;
			ui n=cal(a.size()+g3.a.size()-1),i;
			Q f,g1,g2;
			g1=g2=g3;
			ll x;
			a.resize(n);g1.a.resize(n);g2.a.resize(n);g3.a.resize(n);
			g1.dft1();g2.dft2();g3.dft3();
			f=*this;f.dft1();for (i=0;i<n;i++) g1.a[i]=(ll)g1.a[i]*f.a[i]%p1;g1.dft1(1);
			f=*this;f.dft2();for (i=0;i<n;i++) g2.a[i]=(ll)g2.a[i]*f.a[i]%p2;g2.dft2(1);
			f=*this;f.dft3();for (i=0;i<n;i++) g3.a[i]=(ll)g3.a[i]*f.a[i]%p3;g3.dft3(1);
			a.resize(n>>=1);ui _p12=(ll)p1*p2%p;
			for (i=0;i<n;i++)
			{
				x=(ll)(g2.a[i]+p2-g1.a[i])*inv_p1%p2*p1+g1.a[i];
				a[i]=((x+p3-g3.a[i])%p3*(p3-inv_p12)%p3*_p12+x)%p;
			}
			for (i=n-1;i>=2;i--) if (a[i]) break;
			a.resize(cal(i+1));
		}
	#else
		void operator*=(Q f)
		{
			ui n=cal(a.size()+f.a.size()-1);
			a.resize(n);f.a.resize(n);
			(*this).dft();f.dft();
			for (ui i=0;i<n;i++) a[i]=(ll)a[i]*f.a[i]%p;
			(*this).dft(1);
			ui i;
			for (i=n-1;i>=2;i--) if (a[i]) break;
			a.resize(cal(i+1));
		}
	#endif
#else
	#ifdef MTT
		void operator*=(Q g3)
		{
			Q f,g1,g2;
			g1=g2=g3;
			assert(a.size()==g3.a.size());
			ui n=a.size()<<1,i;
			ll x;
			a.resize(n);g1.a.resize(n);g2.a.resize(n);g3.a.resize(n);
			g1.dft1();g2.dft2();g3.dft3();
			f=*this;f.dft1();for (i=0;i<n;i++) g1.a[i]=(ll)g1.a[i]*f.a[i]%p1;g1.dft1(1);
			f=*this;f.dft2();for (i=0;i<n;i++) g2.a[i]=(ll)g2.a[i]*f.a[i]%p2;g2.dft2(1);
			f=*this;f.dft3();for (i=0;i<n;i++) g3.a[i]=(ll)g3.a[i]*f.a[i]%p3;g3.dft3(1);
			a.resize(n>>=1);ui _p12=(ll)p1*p2%p;
			for (i=0;i<n;i++)
			{
				x=(ll)(g2.a[i]+p2-g1.a[i])*inv_p1%p2*p1+g1.a[i];
				a[i]=((x+p3-g3.a[i])%p3*(p3-inv_p12)%p3*_p12+x)%p;
			}
		}//三模，板子 OJ 5e5 0.9s
	#else
		void operator|=(Q f)//直接卷积，不 shift
		{
			ui n=cal(a.size()+f.a.size()-1);
			a.resize(n);f.a.resize(n);
			dft();f.dft();
			for (ui i=0;i<n;i++) a[i]=(ll)a[i]*f.a[i]%p;
			dft(1);
		}
		Q operator|(Q f) const {f|=*this;return f;}
		void operator*=(Q f)//群内卷积
		{
			assert(a.size()==f.a.size());
			ui n=a.size()<<1;
			a.resize(n);f.a.resize(n);
			dft();f.dft();
			for (ui i=0;i<n;i++) a[i]=(ll)a[i]*f.a[i]%p;
			dft(1);a.resize(n>>1);
		}
		void operator&=(const Q &f)//卷积并 shift
		{
			*this|=f;
			int n=a.size(),i;
			for (i=n-1;i>=2;i--) if (a[i]) break;
			a.resize(cal(i+1));
		}
		Q operator&(Q f) const {f&=*this;return f;}
		void operator^=(Q f)//差卷积
		{
			assert(a.size()==f.a.size());
			ui n=a.size();
			reverse(all(f.a));
			*this|=f;
			copy(a.data()+n-1,a.data()+n*2-1,a.data());
			a.resize(n);
		}
		Q operator^(const Q &f) const {Q g=*this;g^=f;return g;}
	#endif
#endif
	#ifdef MTT
		Q operator~()
		{
			Q q=(*this),r(2);
			ui n=a.size()<<1,i,j,k;a.resize(n);
			r.a[0]=ksm(a[0],p-2);
			for (j=2;j<=n;j<<=1)
			{
				k=j>>1;
				r.a.resize(j);
				q.a.resize(j);
				for (i=0;i<k;i++) q.a[i]=a[i];
				r=-(q*r-2)*r;
				r.a.resize(k);
			}
			n>>=1;
			a.resize(n);
			return r;
		}//trivial
	#else
		Q operator~()
		{
			Q q=(*this),r(2),g(2);
			ui n=a.size(),i,j,k;
			r.a[0]=ksm(a[0],p-2);
			for (j=2;j<=n;j<<=1)
			{
				k=j>>1;
				r.a.resize(j);
				g=r;
				q.a.resize(j);
				for (i=0;i<j;i++) q.a[i]=a[i];
				r.dft();q.dft();
				for (i=0;i<j;i++) q.a[i]=(ll)q.a[i]*r.a[i]%p;
				q.dft(1);
				for (i=0;i<k;i++) q.a[i]=0;
				q.dft();
				for (i=0;i<j;i++) r.a[i]=(ll)q.a[i]*r.a[i]%p;
				r.dft(1);
				for (i=0;i<k;i++) r.a[i]=g.a[i];
				for (i=k;i<j;i++) if (r.a[i]) r.a[i]=p-r.a[i];
			}
			return r;
		}//inv(1 6 3 4 9)=(1 998244347 33 998244169 1020)
	#endif
		Q operator/(Q f) const {return (*this)*~f;}
		void operator/=(Q f) {f=~f;(*this)*=f;}
	};
	Q read(int n)
	{
		Q r(n);
		int P=p;
		int x;
		for (int i=0;i<n;i++) cin>>x,r.a[i]=(x%P+P)%P;
		//for (int i=0;i<n;i++) cin>>r.a[i];
		return r;
	}
	void write(Q &r,int n=-1)
	{
		if (n==0) return;
		if (n==-1) n=r.a.size();
		for (int i=0;i<n-1;i++) cout<<r.a[i]<<" ";
		cout<<r.a[n-1]<<endl;
	}
#ifndef MTT
	Q sqr(Q b)
	{
		ui n=b.a.size()<<1,i;
		b.a.resize(n);
		b.dft();
		for (i=0;i<n;i++) b.a[i]=(ll)b.a[i]*b.a[i]%p;
		b.dft(1);
		b.a.resize(n>>1);
		return b;
	}
	vector<Q> cd;
	void cdq(Q &f,Q &g,ui l,ui r)//g_0=1,i*g_i=g_{i-j}*f_j,use for exp_cdq
	{
		ui i,m=l+r>>1,n=r-l,nn=n>>1;
		if (l==0&&r==f.a.size())
		{
			getinv(n-1);
			g.a.resize(n);
			for (i=0;i<n;i++) g.a[i]=0;
			cd.clear();cd.reserve(__builtin_ctz(n));
			Q a(2);
			for (i=2;i<=n;i<<=1)
			{
				a.a.resize(i);
				for (ui j=0;j<i;j++) a.a[j]=f.a[j];
				a.dft();cd.push_back(a);
			}
		}
		if (l+1==r)
		{
			if (l==0) g.a[l]=1; else g.a[l]=(ll)g.a[l]*inv[l]%p;
			return;
		}
		cdq(f,g,l,m);
		Q a=cd[__builtin_ctz(n)-1],b(n);
		for (i=0;i<nn;i++) b.a[i]=g.a[l+i];
		b.dft();
		for (i=0;i<n;i++) a.a[i]=(ll)a.a[i]*b.a[i]%p;
		a.dft(1);
		for (i=m;i<r;i++) if ((g.a[i]+=a.a[i-l])>=p) g.a[i]-=p;
		cdq(f,g,m,r);
	}
	Q exp_cdq(Q f)
	{
		Q g(2);ui n=f.a.size();
		for (ui i=1;i<n;i++) f.a[i]=(ll)f.a[i]*i%p;
		cdq(f,g,0,n);
		return g;
	}
	Q sqrt(Q b)
	{
		Q q(2),f(2),r(2);
		ui n=b.a.size();
		int i,j=n,l;
		for (i=0;i<n;i++) if (b.a[i]) {j=i;break;}
		if (j==n) return b;
		if (j&1) {puts("-1");exit(0);}l=j>>1;
		for (i=0;i<n-j;i++) b.a[i]=b.a[i+j];
		for (i=n-j;i<n;i++) b.a[i]=0;
		r.a[0]=i=mosqrt(b.a[0]);
		assert(i!=-1);
		for (j=2;j<=n;j<<=1)
		{
			r.a.resize(j);
			q=~r;f.a.resize(j<<1);
			for (i=0;i<j;i++) f.a[i]=b.a[i];
			q.a.resize(j<<1);r.a.resize(j<<1);
			q.dft();r.dft();f.dft();
			for (i=0;i<j<<1;i++) if ((r.a[i]=(ll)q.a[i]*((ll)r.a[i]*r.a[i]%p+f.a[i])%p)&1) r.a[i]=r.a[i]+p>>1; else r.a[i]>>=1;
			r.dft(1);
			for (i=j;i<j<<1;i++) r.a[i]=0;
		}
		r.a.resize(n);
		for (i=n-1;i>=l;i--) r.a[i]=r.a[i-l];
		for (i=0;i<l;i++) r.a[i]=0;
		return r;
	}//sqrt(1 8596489 489489 4894 1564 489 35789489)=(1 503420421 924499237 13354513 217017417 707895465 411020414)
#endif
	Q ln(Q b) {return (b.dao()/b).ji();}//ln(1 927384623 878326372 3882 273455637 998233543)=(0 927384623 817976920 427326948 149643566 610586717)
#ifdef MTT
	Q exp(Q f)
	{
		Q q(2),r(2);
		ui n=f.a.size()<<1,i,j,k;
		r.a[0]=1;
		for (j=2;j<=n;j<<=1)
		{
			k=j>>1;
			r.a.resize(j);
			q.a.resize(j);
			for (i=0;i<k;i++) q.a[i]=f.a[i];
			r=r*(q-ln(r)+1);
			r.a.resize(k);
		}
		return r;
	}
#else
	Q exp(Q b)
	{
		Q q(2),r(2);
		ui n=b.a.size(),i,j;
		r.a[0]=1;
		for (j=2;j<=n;j<<=1)
		{
			r.a.resize(j);
			q=ln(r);
			for (i=0;i<j;i++) if ((q.a[i]=b.a[i]+p-q.a[i])>=p) q.a[i]-=p;
			if (++q.a[0]==p) q.a[0]=0;
			r.a.resize(j<<1);q.a.resize(j<<1);
			r.dft();q.dft();
			for (i=0;i<j<<1;i++) r.a[i]=(ll)r.a[i]*q.a[i]%p;
			r.dft(1);
			r.a.resize(j);
		}
		return r;
	}//exp(0 927384623 817976920 427326948 149643566 610586717)=(1 927384623 878326372 3882 273455637 998233543)
	void mul(Q &a,Q &b)
	{
		ui n=a.a.size();
		a.dft();b.dft();
		for (ui i=0;i<n;i++) a.a[i]=(ll)a.a[i]*b.a[i]%p;
		a.dft(1);
	}
	Q exp_new(Q b)
	{
		Q h(2),f(2),r(2),u(2),v(2);
		ui n=b.a.size(),i,j,k;
		r.a[0]=1;h.a[0]=1;
		for (j=2;j<=n;j<<=1)
		{
			f.a.resize(j);
			for (i=0;i<j;i++) f.a[i]=b.a[i];
			f=f.dao();
			k=j>>1;
			for (i=0;i<k-1;i++) {if ((f.a[i+k]+=f.a[i])>=p) f.a[i+k]-=p;f.a[i]=0;}
			for (i=k-1;i<j;i++) if (f.a[i]) f.a[i]=p-f.a[i];
			u.a.resize(k);v.a.resize(k);
			for (i=0;i<k;i++) u.a[i]=r.a[i],v.a[i]=h.a[i];
			u=u.dao();
			mul(u,v);
			for (i=0;i<k-1;i++) if ((f.a[i+k]+=u.a[i])>=p) f.a[i+k]-=p;
			if ((f.a[k-1]+=u.a[k-1])>=p) f.a[k-1]-=p;
			for (i=0;i<k;i++) u.a[i]=r.a[i];
			u.dft();
			for (i=0;i<k;i++) u.a[i]=(ll)u.a[i]*v.a[i]%p;
			u.dft(1);
			if (u.a[0]) --u.a[0]; else u.a[0]=p-1;
			u.a.resize(j);v.a.resize(j);
			for (i=0;i<k;i++) v.a[i]=b.a[i];
			v=v.dao();
			mul(u,v);
			for (i=0;i<k;i++) if (f.a[i+k]<u.a[i]) f.a[i+k]+=p-u.a[i]; else f.a[i+k]-=u.a[i];
			f=f.ji();
			for (i=0;i<k;i++) u.a[i]=r.a[i];
			for (i=k;i<j;i++) u.a[i]=0;
			mul(u,f);
			r.a.resize(j);
			for (i=k;i<j;i++) if (u.a[i]) r.a[i]=p-u.a[i]; else r.a[i]=0;
			if (j!=n) h=~r;
		}
		return r;
	}
	Q sqrt_new(Q b)
	{
		Q q(2),r(2),h(2);
		ui n=b.a.size();
		int i,j=n,k,l;
		for (i=0;i<n;i++) if (b.a[i]) {j=i;break;}
		if (j==n) return b;
		if (j&1) {puts("-1");exit(0);}l=j>>1;
		for (i=0;i<n-j;i++) b.a[i]=b.a[i+j];
		for (i=n-j;i<n;i++) b.a[i]=0;
		r.a[0]=mosqrt(b.a[0]);h.a[0]=ksm(r.a[0],p-2);
		r.a.resize(1);ui i2=ksm(2,p-2);
		for (j=2;j<=n;j<<=1)
		{
			k=j>>1;
			q=r;
			q.dft();
			for (i=0;i<k;i++) q.a[i]=(ll)q.a[i]*q.a[i]%p;
			q.dft(1);
			q.a.resize(j);
			for (i=k;i<j;i++) q.a[i]=(ll)(q.a[i-k]+p*2u-b.a[i]-b.a[i-k])*i2%p;
			for (i=0;i<k;i++) q.a[i]=0;
			h.a.resize(j);
			mul(q,h);
			r.a.resize(j);
			for (i=k;i<j;i++) if (q.a[i]) r.a[i]=p-q.a[i];
			if (j!=n) h=~r;
		}
		r.a.resize(n);
		for (i=n-1;i>=l;i--) r.a[i]=r.a[i-l];
		for (i=0;i<l;i++) r.a[i]=0;
		return r;
	}
	Q pow(Q b,ui m)
	{
		ui n=b.a.size();
		int i,j=n,k;
		for (i=0;i<n;i++) if (b.a[i]) {j=i;break;}
		if (j==n) return b;
		if ((ll)j*m>=n)
		{
			for (j=0;j<n;j++) b.a[j]=0;
			return b;
		}
		for (i=0;i<n-j;i++) b.a[i]=b.a[i+j];
		for (i=n-j;i<n;i++) b.a[i]=0;
		k=b.a[0];assert(k);
		b=exp_new(ln(b*ksm(k,p-2))*m)*ksm(k,m);
		j*=m;
		for (i=n-1;i>=j;i--) b.a[i]=b.a[i-j];
		for (i=0;i<j;i++) b.a[i]=0;
		return b;
	}
	Q pow2(Q b,ui m)
	{
		Q r(b.a.size());r.a[0]=1;
		while (m)
		{
			if (m&1) r=r*b;
			if (m>>=1) b=b*b;
		}
		return r;
	}
	pair<Q,Q> div(Q a,Q b)
	{
		int n=0,m=0,l,i,nn=a.a.size();
		for (i=a.a.size()-1;i>=0;i--) if (a.a[i]) {n=i+1;break;}
		for (i=b.a.size()-1;i>=0;i--) if (b.a[i]) {m=i+1;break;}
		assert(m);
		if (n<m) return make_pair(Q(2),a);
		l=cal(n+m-1);
		Q c(n),d(m);
		reverse_copy(a.a.data(),a.a.data()+n,c.a.data());
		reverse_copy(b.a.data(),b.a.data()+m,d.a.data());
		c.a.resize(cal(n-m+1));d.a.resize(c.a.size());
		d=~d;
		for (i=n-m+1;i<c.a.size();i++) c.a[i]=d.a[i]=0;
		c*=d;
		for (i=n-m+1;i<c.a.size();i++) c.a[i]=0;
		reverse(c.a.data(),c.a.data()+n-m+1);
		n=a.a.size();b.a.resize(n);c.a.resize(n);
		d=a-c*b;
		//for (i=0;i<d.a.size();i++) cerr<<d.a[i]<<" \n"[i==d.a.size()-1];
		for (i=m;i<d.a.size();i++) assert(d.a[i]==0);
		c.a.resize(cal(n-m+1));d.a.resize(cal(m));
		return make_pair(c,d);
	}
	Q sin(Q &f) {return (exp(f*I)-exp(f*(p-I)))*ksm(2*I%p,p-2);}
	Q cos(Q &f) {return (exp(f*I)+exp(f*(p-I)))*ksm(2,p-2);}
	Q tan(Q &f) {return sin(f)/cos(f);}
	Q asin(Q &f) {return (f.dao()/sqrt((f*f-1)*(p-1))).ji();}
	Q acos(Q &f) {return ((f.dao()/sqrt((f*f-1)*(p-1)))*(p-1)).ji();}
	Q atan(Q &f) {return (f.dao()/(f*f+1)).ji();}
	Q cdq_inv(Q &f) {return (~(f-1))*(p-1);}//g_0=1,g_i=g_{i-j}*f_j
	Q operator%(Q f,Q g) {return div(f,g).second;}
	void operator%=(Q &f,const Q &g) {f=f%g;}
	ui dt(const vector<ui>& f,const vector<ui> &a,ll m)//常系数齐次线性递推，find a_m,a_n=a_{n-i}*f_i,f_1...k,a_0...k-1
	{
		if (m<a.size()) return a[m];
		assert(f.size()==a.size()+1);
		int k=a.size();
		ui n=cal(k+1<<1),i,ans=0,l;
		Q h(n);
		for (i=1;i<=k;i++) if (f[i]) h.a[k-i]=p-f[i];
		h.a[k]=1;
		Q g,r;g.a[1]=1;r.a[0]=1;
		while (m)
		{
			if (m&1) r=(r&g)%h;
			l=g.a.size()<<1;g.a.resize(l);
			g.dft();
			for (i=0;i<l;i++) g.a[i]=(ll)g.a[i]*g.a[i]%p;
			g.dft(1);g%=h;m>>=1;
		}
		k=min(k,(int)r.a.size());
		for (i=0;i<k;i++) ans=(ans+(ll)a[i]*r.a[i])%p;
		return ans;
	}//板子 OJ 1e5/1e18 8246ms，Luogu 32000/1e9 710ms
	ui new_dt(const vector<ui>& f,const vector<ui> &a,ll m)//常系数齐次线性递推，find a_m,a_n=a_{n-i}*f_i,f_1...k,a_0...k-1
	{
		if (m<a.size()) return a[m];
		assert(f.size()==a.size()+1);
		ui k=a.size(),n=cal(k+1),lim=n*2,x;
		int i;
		Q g(n),h(n);
		vector<ui> res(k);
		for (i=1;i<=k;i++) if (f[i]) h.a[i]=p-f[i];
		h.a[0]=1;
		for (i=0;i<k;i++) g.a[i]=a[i];
		g*=h;fill(g.a.data()+k,g.a.data()+n,0);
		++k;g.a.resize(lim);h.a.resize(lim);
		while (m)
		{
			if (m&1)
			{
				x=p-g.a[0];
				for (i=1;i<k;i+=2) res[i>>1]=(ll)x*h.a[i]%p;
				copy(g.a.data()+1,g.a.data()+k,g.a.data());
				g.a[k-1]=0;
			}
			g.dft();h.dft();
			ui *a=g.a.data(),*b=h.a.data(),*c=a+n,*d=b+n;
			for (i=0;i<n;i++) g.a[i]=((ll)a[i]*d[i]+(ll)b[i]*c[i]+(ll)((a[i]&d[i])^(b[i]&c[i]))*p>>1)%p;
			for (i=0;i<n;i++) h.a[i]=(ll)h.a[i]*h.a[i^n]%p;
			g.hf_dft(1);h.hf_dft(1);
			fill(g.a.data()+k,g.a.data()+lim,0);
			if (m&1) for (i=0;i<k;i++) if ((g.a[i]+=res[i])>=p) g.a[i]-=p;
			fill(h.a.data()+k,h.a.data()+lim,0);
			m>>=1;
		}
		return g.a[0];
	}//板子 OJ 1e5/1e18 1310ms，Luogu 32000/1e9 160ms
#endif
#ifdef normal
	Q mult(Q *a,int n)
	{
		if (n==1) return a[0];
		int m=n>>1;
		return mult(a,m)&mult(a+m,n-m);
	}
#endif
	vector<Q> pro;
	ui *X;
	vector<ui> Y;
	void build(int x,int l,int r)
	{
		if (l==r)
		{
			pro[x].a.resize(2);pro[x].a[0]=(p-X[l])%p;
			pro[x].a[1]=1;
			return;
		}
		int mid=l+r>>1,c=x<<1;
		build(c,l,mid);build(c|1,mid+1,r);
		pro[x]=pro[c]&pro[c|1];
	}
	void sgt_dfs(int x,int l,int r,Q f,int d)
	{
		if (d>=r-l+1)
		{
			f%=pro[x];
			d=r-l;
			while (d>0&&!f.a[d]) --d;
			f.a.resize(cal(d+1));
		}
		if (r-l+1<=255)
		{
			for (ui i=l;i<=r;i++) Y[i]=f.fx(X[i]);
			return;
		}
		int mid=l+r>>1,c=x<<1;
		sgt_dfs(c,l,mid,f,d);
		sgt_dfs(c|1,mid+1,r,f,d);
	}
	vector<ui> get_fx(Q &f,vector<ui> &x)
	{
		int m=x.size(),i,j;
		int n=f.a.size()-1;
		pro.resize(m*4+8);
		while (n>1&&!f.a[n]) --n;
		X=x.data();Y.resize(m);
		build(1,0,m-1);
		sgt_dfs(1,0,m-1,f,n);
		return Y;
	}
	void new_build(int x,int n)
	{
		if (n==1)
		{
			pro[x].a.resize(2);pro[x].a[0]=1;
			pro[x].a[1]=(p-*(X++))%p;
			return;
		}
		int mid=n+1>>1,c=x<<1;
		new_build(c,mid);new_build(c|1,n-mid);
		pro[x]=pro[c]&pro[c|1];
	}
	const int get_fx_lim2=30;
	void new_sgt_dfs(int x,int l,int r,Q f)
	{
		if (r-l+1<=get_fx_lim2)
		{
			int m=r-l+1,m1,m2,mid=l+r>>1,i,j,k;
			static ui g[get_fx_lim2+2],g1[get_fx_lim2+2],g2[get_fx_lim2+2];
			m1=m2=r-l;
			memcpy(g1,f.a.data(),m*sizeof g1[0]);
			memcpy(g2,f.a.data(),m*sizeof g2[0]);
			for (i=mid+1;i<=r;i++) {for (k=0;k<m1;k++) g1[k]=(g1[k]+(ll)g1[k+1]*(p-X[i]))%p;--m1;}
			for (i=l;i<=mid;i++) {for (k=0;k<m2;k++) g2[k]=(g2[k]+(ll)g2[k+1]*(p-X[i]))%p;--m2;}
			for (i=l;i<=mid;i++)
			{
				m=m1;
				memcpy(g,g1,(m1+1)*sizeof g[0]);
				for (j=l;j<=mid;j++) if (i!=j)
				{
					for (k=0;k<m;k++) g[k]=(g[k]+(ll)g[k+1]*(p-X[j]))%p;
					--m;
				}
				Y[i]=g[0];
			}
			for (i=mid+1;i<=r;i++)
			{
				m=m2;
				memcpy(g,g2,(m2+1)*sizeof g[0]);
				for (j=mid+1;j<=r;j++) if (i!=j)
				{
					for (k=0;k<m;k++) g[k]=(g[k]+(ll)g[k+1]*(p-X[j]))%p;
					--m;
				}
				Y[i]=g[0];
			}
			return;
		}
		int mid=l+r>>1,c=x<<1,n=f.a.size();
		pro[c].a.resize(n);
		pro[c|1].a.resize(n);
		f.dft();reverse(all(pro[c].a));
		pro[c].dft();
		for (int i=0;i<n;i++) pro[c].a[i]=(ll)pro[c].a[i]*f.a[i]%p;
		pro[c].dft(1);rotate(all(pro[c].a)-1,pro[c].a.end());
		pro[c].a.resize(cal(r-mid));fill(pro[c].a.begin()+r-mid,pro[c].a.end(),0);
		c^=1;
		reverse(all(pro[c].a));
		pro[c].dft();
		for (int i=0;i<n;i++) pro[c].a[i]=(ll)pro[c].a[i]*f.a[i]%p;
		pro[c].dft(1);rotate(all(pro[c].a)-1,pro[c].a.end());
		pro[c].a.resize(cal(mid-l+1));fill(pro[c].a.begin()+mid-l+1,pro[c].a.end(),0);
		c^=1;
		new_sgt_dfs(c,l,mid,pro[c|1]);
		new_sgt_dfs(c|1,mid+1,r,pro[c]);
	}
	vector<ui> new_get_fx(Q f,vector<ui> &x)//多项式多点求值
	{
		if (x.size()==0) return x;
		int m=x.size(),i,j;
		if (x.size()<=10)
		{
			Y.resize(m);
			for (i=0;i<m;i++) Y[i]=f.fx(x[i]);
			return Y;
		} 
		int n=f.a.size();
		while (n>1&&!f.a[n-1]) --n;
		if (cal(n)!=f.a.size()) f.a.resize(cal(n));
		X=x.data();Y.resize(m);
		pro.resize(m*4+8);
		new_build(1,m);X=x.data();
		pro[1].a.resize(f.a.size());
		f^=~pro[1];
		f.a.resize(cal(m));
		fill(f.a.begin()+min(m,n),f.a.end(),0);
		new_sgt_dfs(1,0,m-1,f);
		return Y;
	}//板子 OJ 2^17 550ms
	vector<Q> sum;
	void get_poly_build(int x,int n)
	{
		if (n==1)
		{
			sum[x].a.resize(2);
			sum[x].a[1]=1;
			sum[x].a[0]=(p-*(X++))%p;
			return;
		}
		int mid=n+1>>1,c=x<<1;
		get_poly_build(c,mid);get_poly_build(c|1,n-mid);
		sum[x]=sum[c]&sum[c|1];
	}
	void get_poly_dfs(int x,int l,int r)
	{
		if (l==r)
		{
			pro[x].a.resize(2);
			pro[x].a[0]=Y[l];
			pro[x].a[1]=0;
			return;
		}
		int c=x<<1,mid=l+r>>1;
		get_poly_dfs(c,l,mid);get_poly_dfs(c|1,mid+1,r);
		pro[x]=(pro[c]&sum[c|1])+(pro[c|1]&sum[c]);
	}
	Q get_poly(vector<ui> &x,vector<ui> &y)//多项式快速插值
	{
		assert(x.size()==y.size());
		int n=x.size(),i,j;
		if (n==0) return Q(2);
		if (n==1)
		{
			Q f(2);
			f.a[0]=y[0];
			return f;
		}
		if (1)
		{
			auto vv=x;sort(all(vv));
			assert(unique(all(vv))-vv.begin()==n);
		}
		sum.resize(4*n+8);X=x.data();
		get_poly_build(1,n);
		sum[1]=sum[1].dao();
		auto v=new_get_fx(sum[1],x);
		pro.resize(4*n+8);
		assert(v.size()==n);
		Y=getinvs(v);
		for (i=0;i<n;i++) Y[i]=(ll)Y[i]*y[i]%p;
		get_poly_dfs(1,0,n-1);
		pro[1].a.resize(cal(n));
		return pro[1];
	}//板子 OJ 2^17 1.3s
	Q comp(const Q &f,Q g)//多项式复合 f(g(x))=[x^i]f(x)g(x)^i
	{
		int n=f.a.size(),l=ceil(::sqrt(n)),i,j;
		assert(n>=g.a.size());//返回 n-1 次多项式
		vector<Q> a(l+1),b(l);
		a[0].a.resize(n);a[0].a[0]=1;a[1]=g;
		g.a.resize(n*2);
		Q w=g,u,v(n);
		w.dft();u=w;
		for (i=2;i<=l;i++)
		{
			if (i>2) u.dft();
			for (j=0;j<n*2;j++) u.a[j]=(ll)u.a[j]*w.a[j]%p;
			u.dft(1);
			fill(u.a.data()+n,u.a.data()+n*2,0);
			a[i]=u;
		}
		w=a[l];
		w.dft();u=w;
		for (i=2;i<=l;i++) a[i].a.resize(n);
		for (i=2;i<l;i++)
		{
			if (i>2) u.dft();
			b[i-1]=u;
			for (j=0;j<n*2;j++) u.a[j]=(ll)u.a[j]*w.a[j]%p;
			u.dft(1);
			fill(u.a.data()+n,u.a.data()+n*2,0);
		}
		if (l>2) u.dft();b[l-1]=u;
		for (i=0;i<l;i++)
		{
			fill(all(v.a),0);
			for (j=0;j<l;j++) if (i*l+j<n) v+=a[j]*f.a[i*l+j];
			if (i==0) u=v; else
			{
				v.a.resize(n*2);v.dft();
				for (j=0;j<n*2;j++) v.a[j]=(ll)v.a[j]*b[i].a[j]%p;
				v.dft(1);v.a.resize(n);u+=v;
			}
		}
		return u;
	}//n^2+n\sqrt n\log n，8000 板子 OJ 300ms，20000 luogu 3.5s
	Q comp_inv(Q f)//多项式复合逆 g(f(x))=x，求 g，[x^n]g=([x^{n-1}](x/f)^n)/n，要求常数 0 一次非 0
	{
		assert(!f.a[0]&&f.a[1]);
		int n=f.a.size(),l=ceil(::sqrt(n)),i,j,k,m;//l>=2
		for (i=1;i<n;i++) f.a[i-1]=f.a[i];f.a[n-1]=0;
		f=~f;
		getinv(n*2);
		vector<Q> a(l+1),b(l);
		Q u,v;
		u=a[1]=f;
		u.a.resize(n*2);u.dft();v=u;
		for (i=2;i<=l;i++)
		{
			if (i>2) u.dft();
			for (j=0;j<n*2;j++) u.a[j]=(ll)u.a[j]*v.a[j]%p;
			u.dft(1);fill(u.a.data()+n,u.a.data()+n*2,0);
			a[i]=u;
		}
		b[0].a.resize(n);b[0].a[0]=1;b[1]=a[l];u.dft();v=u;
		for (i=2;i<l;i++)
		{
			if (i>2) u.dft();
			for (j=0;j<n*2;j++) u.a[j]=(ll)u.a[j]*v.a[j]%p;
			u.dft(1);fill(u.a.data()+n,u.a.data()+n*2,0);
			b[i]=u;
		}
		u.a.resize(n);u.a[0]=0;
		for (i=0;i<l;i++) for (j=1;j<=l;j++) if (i*l+j<n)
		{
			m=i*l+j-1;
			ui r=0,*f=b[i].a.data(),*g=a[j].a.data();
			for (k=0;k<=m;k++) r=(r+(ll)f[k]*g[m-k])%p;
			u.a[m+1]=(ll)r*inv[m+1]%p;
		}
		return u;
	}
	Q shift(Q f,ui c)//get f(x+c)
	{
		int n=f.a.size(),i,j;
		Q g(n);
		getfac(n);
		for (i=0;i<n;i++) f.a[i]=(ll)f.a[i]*fac[i]%p;
		g.a[0]=1;
		for (i=1;i<n;i++) g.a[i]=(ll)g.a[i-1]*c%p;
		for (i=0;i<n;i++) g.a[i]=(ll)g.a[i]*ifac[i]%p;
		f^=g;
		for (i=0;i<n;i++) f.a[i]=(ll)f.a[i]*ifac[i]%p;
		return f;
	}
	vector<ui> point_shift(vector<ui> y,ui c,ui m)//[0,n) 点值 -> [c,c+m) 点值
	{
		assert(y.size());
		if (y.size()==1) return vector<ui>(m,y[0]);
		vector<ui> r,res;
		r.reserve(m);
		int n=y.size(),i,j,mm=m;
		while (c<n&&m) r.push_back(y[c++]),--m;
		if (c+m>p)
		{
			res=point_shift(y,0,c+m-p);
			m=p-c;
		}
		if (!m) {r.insert(r.end(),all(res));return r;}
		int len=cal(m+n-1),l=m+n-1;
		for (i=n&1;i<n;i+=2) if (y[i]) y[i]=p-y[i];
		getfac(n);
		for (i=0;i<n;i++) y[i]=(ll)y[i]*ifac[i]%p*ifac[n-1-i]%p;
		y.resize(len);
		Q f,g;
		vector<ui> v(m+n-1);
		c-=n-1;
		for (i=0;i<l;i++) v[i]=(c+i)%p;
		f.a=y;g.a=getinvs(v);g.a.resize(len);
		f|=g;
		vector<ui> u(m);
		for (i=n-1;i<l;i++) u[i-(n-1)]=f.a[i];
		v.resize(m);
		for (i=0;i<m;i++) v[i]=c+i;
		v=getinvs(v);c+=n;
		ui tmp=1;
		for (i=c-n;i<c;i++) tmp=(ll)tmp*i%p;
		for (i=0;i<m;i++) u[i]=(ll)u[i]*tmp%p,tmp=(ll)tmp*(c+i)%p*v[i]%p;
		r.insert(r.end(),all(u));
		r.insert(r.end(),all(res));
		assert(r.size()==mm);
		return r;
	}//板子 OJ 20w 370ms，luogu 16w 150ms
	const ui B=1e5;
	ui a[B+2],b[B+2];
	ui mic(ui x) {return (ll)a[x%B]*b[x/B]%p;}
	vector<ui> Z_trans(Q f,ui c,ui m)//求 f(c^[0,m))。核心 ij=C(i+j,2)-C(i,2)-C(j,2)
	{
		ui i,j,n=f.a.size();
		if ((ll)n*m<B*5)
		{
			vector<ui> r(m);
			for (i=0,j=1;i<m;i++) r[i]=f.fx(j),j=(ll)j*c%p; 
			return r;
		}
		ui l=cal(m+=n-1);
		Q g(l);
		assert((ll)B*B>p);
		a[0]=b[0]=g.a[0]=g.a[1]=1;
		for (i=1;i<=B;i++) a[i]=(ll)a[i-1]*c%p;
		c=a[B];
		for (i=1;i<=B;i++) b[i]=(ll)b[i-1]*c%p;
		for (i=2;i<n;i++) f.a[i]=(ll)f.a[i]*mic((ll)(p*2-2-i)*(i-1)/2%(p-1))%p;
		reverse(all(f.a));f.a.resize(l);
		for (i=2;i<m;i++) g.a[i]=mic((ll)i*(i-1)/2%(p-1));
		f.dft();g.dft();
		for (i=0;i<l;i++) f.a[i]=(ll)f.a[i]*g.a[i]%p;
		f.dft(1);
		vector<ui> r(f.a.data()+n-1,f.a.data()+m);m-=n-1;
		for (i=2;i<m;i++) r[i]=(ll)r[i]*mic((ll)(p*2-2-i)*(i-1)/2%(p-1))%p;
		return r;
	}//luogu 1e6 400ms
	vector<ui> get_Bell(int n)//B(0...n)
	{
		++n;
		getfac(n-1);
		Q f(n);
		int i;
		for (i=1;i<n;i++) f.a[i]=ifac[i];
		f=exp(f);
		for (i=2;i<n;i++) f.a[i]=(ll)f.a[i]*fac[i]%p;
		return vector<ui>(f.a.data(),f.a.data()+n);
	}
	vector<ui> get_S1_row(int n,int m)//S1(n,0...m),O(nlogn),unsigned
	{
		int cm=cal(++m);
		if (n==0)
		{
			vector<ui> r(m);
			r[0]=1;
			return r;
		}
		auto dfs=[&](auto self,int n)
		{
			if (n==1)
			{
				Q f(2);
				f.a[1]=1;
				return f;
			}
			Q f=self(self,n>>1);
			f|=shift(f,n>>1);
			if (n&1)
			{
				f.a.resize(cal(n+1));
				copy_n(f.a.data(),n,f.a.data()+1);
				--n;
				for (ui i=0;i<=n;i++) f.a[i]=(f.a[i]+(ll)f.a[i+1]*n)%p;
			}
			if (f.a.size()>cm) f.a.resize(cm);
			return f;
		};
		Q f=dfs(dfs,n);
		return vector<ui>(f.a.data(),f.a.data()+m);
	}
	vector<ui> get_S1_column(int n,int m)//S1(0...n,m),O(nlogn)
	{
		if (m==0)
		{
			vector<ui> r(n+1);
			r[0]=1;
			return r;
		}
		Q f(n+1);
		getfac(max(n,m));
		int i;
		for (i=1;i<=n;i++) f.a[i]=inv[i];
		f=pow(f,m);
		for (i=m;i<=n;i++) f.a[i]=(ll)f.a[i]*fac[i]%p*ifac[m]%p;
		return vector<ui>(f.a.data(),f.a.data()+n+1);
	}
	vector<ui> get_S2_row(int n,int m)//S2(n,0...m),O(mlogm)
	{
		int tm=++m;
		if (n==0)
		{
			vector<ui> r(m);
			r[0]=1;
			return r;
		}
		m=min(m,n+1);
		ui pr[m],pw[m],cnt=0;
		int i,j;
		fill_n(pw,m,0);
		pw[1]=1;
		for (i=2;i<m;i++)
		{
			if (!pw[i]) pr[cnt++]=i,pw[i]=ksm(i,n);
			for (j=0;i*pr[j]<m;j++)
			{
				pw[i*pr[j]]=(ll)pw[i]*pw[pr[j]]%p;
				if (i%pr[j]==0) break;
			}
		}
		getfac(m-1);
		Q f(m),g(m);
		for (i=0;i<m;i+=2) f.a[i]=ifac[i];
		for (i=1;i<m;i+=2) f.a[i]=p-ifac[i];
		for (i=1;i<m;i++) g.a[i]=(ll)pw[i]*ifac[i]%p;
		f*=g;
		vector<ui> r(f.a.data(),f.a.data()+m);
		r.resize(tm);
		return r;
	}
	vector<ui> get_S2_column(int n,int m)//S2(0...n,m),O(nlogn)
	{
		if (m==0)
		{
			vector<ui> r(n+1);
			r[0]=1;
			return r;
		}
		Q f(n+1);
		getfac(max(n,m));
		int i;
		for (i=1;i<=n;i++) f.a[i]=ifac[i];
		f=pow(f,m);
		for (i=m;i<=n;i++) f.a[i]=(ll)f.a[i]*fac[i]%p*ifac[m]%p;
		return vector<ui>(f.a.data(),f.a.data()+n+1);
	}
	vector<ui> get_signed_S1_row(int n,int m)
	{
		auto v=get_S1_row(n,m);
		for (ui i=1^n&1;i<=m;i+=2) if (v[i]) v[i]=p-v[i];
		return v;
	}
	vector<ui> get_Bernoulli(int n)//B(0...n)
	{
		getfac(++n);
		int i;
		Q f(n);
		for (i=0;i<n;i++) f.a[i]=ifac[i+1];
		f=~f;
		for (i=0;i<n;i++) f.a[i]=(ll)f.a[i]*fac[i]%p;
		return vector<ui>(f.a.data(),f.a.data()+n);
	}
	vector<ui> get_partition(int n)//P(0...n)，分拆数
	{
		Q f(++n);
		int i,l=0,r=0;
		while (--l) if (3*l*l-l>=n*2) break;
		while (++r) if (3*r*r-r>=n*2) break;
		++l;
		for (i=l+abs(l)%2;i<r;i+=2) f.a[3*i*i-i>>1]=1;
		for (i=l+abs(l+1)%2;i<r;i+=2) f.a[3*i*i-i>>1]=p-1;
		f=~f;
		return vector<ui>(f.a.data(),f.a.data()+n);
	}
}
using NTT::ui;using NTT::read;using NTT::p;
#define poly NTT::Q
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	int n,m;
	cin>>n;
	auto v=NTT::get_partition(n);
	for (ui x:v) cout<<x<<' ';
}
```

#### FFT

```cpp
struct fs
{
	double x,y;
};//x+yi
int n,m,i,j,r[N],limit,l;
fs a[N],b[N];
char c;
void read(double &x)
{
	x=0;c=getchar();
	while ((c<48)||(c>57)) c=getchar();
	while ((c>=48)&&(c<=57))
	{
		x=x*10+c-48;
		c=getchar();
	}
}
fs operator + (fs a,fs b)
{
	fs c;
	c.x=a.x+b.x;
	c.y=a.y+b.y;
	return c;
}
fs operator - (fs a,fs b)
{
	fs c;
	c.x=a.x-b.x;
	c.y=a.y-b.y;
	return c;
}
fs operator * (fs a,fs b)
{
	fs c;
	c.x=a.x*b.x-a.y*b.y;
	c.y=a.x*b.y+a.y*b.x;
	return c;
}
void dft(fs *a,int xs)
{
	int i,j,k;
	fs w,wn,b,c;
	for (i=0;i<limit;i++) if (i<r[i]) swap(a[i],a[r[i]]);
	for (i=1;i<limit;i<<=1)
	{
		wn.x=cos(pi/i);
		wn.y=xs*sin(pi/i);
		l=i<<1;
		for (j=0;j<limit;j+=l)
		{
			w.x=1;w.y=0;
			for (k=0;k<i;k++,w=w*wn)
			{
				b=a[j+k];c=a[i+j+k]*w;
				a[j+k]=b+c;
				a[i+j+k]=b-c;
			}
		}
	}
}
```



#### 万能欧几里得

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef unsigned int ui;
typedef long long ll;
typedef unsigned long long ull;
const ui p=998244353;
int t,A,B,C,n,T;
struct nd
{
	int x,y,sx,sy,syy,sxy;
	nd ()
	{
		x=y=sx=sy=syy=sxy=0;
	}
	nd operator+(const nd &o) const
	{
		nd r;
		if ((r.x=x+o.x)>=p) r.x-=p;
		if ((r.y=y+o.y)>=p) r.y-=p;
		r.sx=(sx+o.sx+(ull)x*o.x)%p;
		r.sy=(sy+o.sy+(ull)y*o.x)%p;
		r.sxy=(sxy+o.sxy+(ull)x*o.sy+(ull)y*o.sx+(ull)x*y%p*o.x)%p;
		r.syy=(syy+o.syy+(ull)y*y%p*o.x+2ull*y*o.sy)%p;
		return r;
	}
}nx,ny,ans;
nd ksm (nd a,int k)
{
	nd res;
	while (k)
	{
		if (k&1) res=res+a;
		a=a+a,k>>=1;
	}
	return res;
}
nd sol (int p,int q,int r,int l,nd a,nd b)//(0,l],(i^P)((pi+r)/q)^Q
{
	if (!l) return nd();
	if (p>=q) return sol(p%q,q,r,l,a,ksm(a,p/q)+b);
	int m=((ll)l*p+r)/q;
	if (!m) return ksm(b,l);
	int cnt=l-((ll)q*m-r-1)/p;
	return ksm(b,(q-r-1)/p)+a+sol(q,p,(q-r-1)%p,m-1,b,a)+ksm(b,cnt);
}
int main ()
{
	ios::sync_with_stdio(0);cin.tie(0);cout.tie(0);
	cin>>T;
	while (T--)
	{
		cin>>n>>A>>B>>C;//(0,n],(i^P)((Ai+B)/C)^Q
		nx.x=1;ny.y=1;nx.sx=1;
		ans=ksm(ny,B/C)+sol(A,C,B%C,n,ny,nx);
		cout<<(ans.sy+B/C)%p<<" "<<(ans.syy+(ull)B/C*(B/C))%p<<" "<<ans.sxy<<"\n";
	}
}
```

<div style="page-break-after:always"></div>

### 字符串

#### hash

$O(n)$，$O(n)$。

```cpp
typedef unsigned int ui;
typedef unsigned long long ull;
typedef pair<ui,ui> pa;
namespace sh
{
	const int N=1e6+2;
	const ull b1=137,b2=149,i1=1'603'801'661,i2=1'024'053'074;
	const ui p1=2'034'452'107,p2=2'013'074'419;
	ull m1[N],m2[N],r1,r2;
	int i;
	void init()
	{
		m1[0]=m2[0]=1;
		for (i=1;i<N;i++)
			m1[i]=m1[i-1]*b1%p1,
			m2[i]=m2[i-1]*b2%p2;
	}
	void cal(int *s,int n,pa *a)
	{
		r1=r2=0;
		a[0]=pa(0,0);
		for (i=1;i<=n;i++)
			r1=(r1+s[i]*m1[i])%p1,
			r2=(r2+s[i]*m2[i])%p2,
			a[i]=pa(r1,r2);
	}
	ull getv(pa *a,int l,int r)
	{
		return (p1+a[r].first-a[l-1].first)*m1[N-l]%p1<<32|(p2+a[r].second-a[l-1].second)*m2[N-l]%p2;
	}
}
using sh::init,sh::cal,sh::getv,sh::p1,sh::p2,sh::m1,sh::m2;
```

#### KMP

$O(n)$，$O(n)$。

```cpp
char s[N],t[N];
int nxt[N],pos[N];
int m,i,n;
void get_next(char *s,int *nxt)
{
	int n,i,j;
	n=strlen(s+1);j=0;
	nxt[1]=0;
	for (i=2;i<=n;i++)
	{
		while (j&&s[i]!=s[j+1]) j=nxt[j];
		j+=s[i]==s[j+1];
		nxt[i]=j;
	}
}
int kmp_match(char *s,char *t,int *nxt,int *pos)//find t in s
{
	int n,m,i,j,cnt;
	n=strlen(s+1);m=strlen(t+1);j=0;cnt=0;
	for (i=1;i<=n;i++)
	{
		while (j&&s[i]!=t[j+1]) j=nxt[j];
		j+=s[i]==t[j+1];
		if (j==m) j=nxt[j],pos[++cnt]=i-m+1;//s[i-m+1,i]=t[1,m]
	}
	return cnt;
}
int main()
{
	cin>>s+1>>t+1;m=strlen(t+1);
	get_next(t,nxt);n=kmp_match(s,t,nxt,pos);
	for (i=1;i<=n;i++) cout<<pos[i]<<"\n";
	for (i=1;i<=m;i++) cout<<nxt[i]<<" \n"[i==m];
}
```

#### SAM

$O(n\sum)$，$O(2n\sum )$。

```cpp
template<int N> struct sam
{
	int p,q,np,nq,c[N*2][26],ds,cd,len[N*2],fa[N*2];
	void csh()
	{
		np=ds=1;
	}
	void ins(int zf)
	{
		p=np;len[np=++ds]=++cd;
		while ((c[p][zf]==0)&&(p))
		{
			c[p][zf]=np;
			p=fa[p];
		}
		if (!p)
		{
			fa[np]=1;
			return;
		}
		q=c[p][zf];
		if (len[q]==len[p]+1)
		{
			fa[np]=q;
			return;
		}
		len[nq=++ds]=len[p]+1;
		memcpy(c[nq],c[q],sizeof(c[q]));
		fa[nq]=fa[q];
		fa[np]=fa[q]=nq;
		c[p][zf]=nq;
		while (c[p=fa[p]][zf]==q) c[p][zf]=nq;
	}
	void out()
	{
		int i,j;
		for (i=1;i<=ds;i++) for (j=0;j<=25;j++) if (c[i][j]) printf("%d->%d %c\n",i,c[i][j],j+'a');
	}
	vector<int> match(string s)
	{
		vector<int> r;
		r.reserve(s.size());
		p=1;
		int nl=0;
		for (auto ch:s)
		{
			ch-='a';
			if (c[p][ch]) ++nl,p=c[p][ch];
			else
			{
				while (p&&c[p][ch]==0) p=fa[p];
				if (p==0) p=1,nl=0; else nl=len[p]+1,p=c[p][ch];
			}
			r.push_back(nl);
		}
		return r;
	}
};
```

#### SqAM

$O(n\sum)$，$O(n\sum )$。

```cpp
struct sqam
{
	int c[N][26],ds,i,j,lst[26],pre[N];
	void csh()
	{
		ds=1;
	}
	void ins(int zf)
	{
		++ds;
		for (i=0;i<=25;i++) if (lst[i]) for (j=lst[i];(j)&&(c[j][zf]==0);j=pre[j]) c[j][zf]=ds;
		if (!lst[zf]) c[1][zf]=ds; else pre[ds]=lst[zf];
		lst[zf]=ds;
	}
};
```

#### 后缀树（ukkonen）

$O(n)$，$O(2n\sum )$。

```cpp
void dfs(int x,int lf)
{
	if (!fir[x])
	{
		siz[x][1]=1;
		return;
	}
	int i,j;
	for (i=fir[x];i;i=nxt[i])
	{
		j=c[x][lj[i]];
		if ((f[j]<=m)&&(t[j]>=m)) ++siz[x][0];
		dfs(zd[j],t[j]-f[j]+1);
		siz[x][0]+=siz[zd[j]][0];
		siz[x][1]+=siz[zd[j]][1];
		if ((t[j]==n)&&(f[j]<=m)) --siz[x][1];
	}
	ans+=(ll)siz[x][0]*siz[x][1]*lf;
}
void add(int a,int b,int cc,int d)
{
	zd[++bbs]=b;
	t[bbs]=d;
	c[a][s[f[bbs]=cc]]=bbs;
}
void add(int x,int y)
{
	lj[++bs]=y;
	nxt[bs]=fir[x];
	fir[x]=bs;
}
	s[++m]=26;
	fa[1]=point=ds=1;
	for (i=1;i<=m;i++)
	{
		ad=0;++remain;
		while (remain)
		{
			if (r==0) edge=i;
			if ((j=c[point][s[edge]])==0)
			{
				fa[++ds]=1;
				fa[ad]=point;
				add(ad=point,ds,edge,m);
				add(point,s[edge]);
			}
			else
			{
				if ((t[j]!=m)&&(t[j]-f[j]+1<=r))
				{
					r-=t[j]-f[j]+1;
					edge+=t[j]-f[j]+1;
					point=zd[j];
					continue;
				}
				if (s[f[j]+r]==s[i]) {++r;fa[ad]=point;break;}
				fa[fa[ad]=++ds]=1;
				add(ad=ds,zd[j],f[j]+r,t[j]);
				add(ds,s[i]);add(ds,s[f[j]+r]);fa[++ds]=1;
				add(ds-1,ds,i,m);
				zd[j]=ds-1;t[j]=f[j]+r-1;
			}
			--remain;
			if ((r)&&(point==1))
			{
				--r;edge=i-remain+1;
			} else point=fa[point];
		}
	}
	for (i=1;i<=ds;i++) for (j=fir[i];j;j=nxt[j]) {len[j]=t[c[i][lj[j]]]-f[c[i][lj[j]]]+1;lj[j]=zd[c[i][lj[j]]];}
```

#### 最小表示法

$O(n)$，$O(1)$。

```cpp
template<typename T> void min_order(T *a,int n)//[0,n)
{
	int i,j,k;
	T x,y;
	i=k=0;j=1;
	while (i<n&&j<n&&k<n)
	{
		x=a[(i+k)%n];y=a[(j+k)%n];
		if (x==y) ++k; else
		{
			if (x>y) i+=k+1; else j+=k+1;
			if (i==j) ++j;
			k=0;
		}
	}
	if (j>i) j=i;
	//[j,n)+[0,j)
	rotate(a,a+j,a+n);
}
```

#### manacher

$O(n)$，$O(n)$。

```cpp
vector<int> manacher(const string &t)//ex[i](total length) centered at i/2
{
	string S="$#";
	int n=t.size(),i,r=1,m=0;
	for (i=0;i<n;i++) S+=t[i],S+='#';
	S+='#';
	char *s=S.data()+2;
	n=n*2-1;
	vector<int> ex(n);
	ex[0]=2;
	for (i=1;i<n;i++)
	{
		ex[i]=i<r?min(ex[m*2-i],r-i+1):1;
		while (s[i+ex[i]]==s[i-ex[i]]) ++ex[i];
		if (i+ex[i]-1>r) r=i+ex[m=i]-1;
	}
	for (i=0;i<n;i++) --ex[i];
	return ex;
}
```

#### AC 自动机

```cpp
scanf("%d",&n);
	for (i=1;i<=n;i++)
	{
		x=0;cc=getchar();
		while ((cc<'a')||(cc>'z')) cc=getchar();
		while ((cc>='a')&&(cc<='z'))
		{
			cc-='a';
			if (c[x][cc]==0) c[x][cc]=++ds;
			x=c[x][cc];
			cc=getchar();
		}
		ys[i]=x;
	}tou=1;wei=0;
	for (i=0;i<=25;i++) if (c[0][i]) dl[++wei]=c[0][i];
	while (tou<=wei)
	{
		x=dl[tou++];
		for (i=0;i<=25;i++) if (c[x][i]) f[dl[++wei]=c[x][i]]=c[f[x]][i]; else c[x][i]=c[f[x]][i];
	}
	x=0;cc=getchar();
	while ((cc<'a')||(cc>'z')) cc=getchar();
	while ((cc>='a')&&(cc<='z'))
	{
		++cs[x=c[x][cc-'a']];cc=getchar();
	}++wei;
	while (--wei) cs[f[dl[wei]]]+=cs[dl[wei]];
	for (i=1;i<=n;i++) printf("%d\n",cs[ys[i]]);
```

#### SA

$O((n+\sum)\log n)$，$O(n+\sum)$。

```cpp
namespace SA
{
	const int N=1e6+2;
	int x[N],y[N],s[N],lg[N];
	int m,i,j,k,cnt;
	bool ied=0;
	void SA_init()
	{
		for (int i=2;i<N;i++) lg[i]=lg[i>>1]+1;
	}
	struct Q
	{
		vector<vector<int>> st;
		vector<int> _sa,rk,h;
		int *sa;
		int lcp(int x,int y)
		{
			assert(x^y);
			x=rk[x];y=rk[y];
			if (x>y) swap(x,y);
			++x;
			int z=lg[y-x+1];
			return min(st[z][x],st[z][y-(1<<z)+1]);
		}
		Q(int *a,int n)//[1,n]
		{
			if (!ied) ied=1,SA_init();
			_sa.resize(n+1);rk.resize(n+1);h.resize(n+1);sa=_sa.data();
			m=*min_element(a+1,a+n+1);--m;
			for (i=1;i<=n;i++) a[i]-=m;
			m=*max_element(a+1,a+n+1);
			assert(n<N);assert(m<N);
			memset(s+1,0,m*sizeof s[0]);
			for (i=1;i<=n;i++) ++s[x[i]=a[i]];
			for (i=2;i<=m;i++) s[i]+=s[i-1];
			for (i=n;i;i--) sa[s[x[i]]--]=i;
			memset(s+1,0,m*sizeof s[0]);
			for (j=1;j<=n;j<<=1)
			{
				cnt=0;
				for (i=n-j+1;i<=n;i++) y[++cnt]=i;
				for (i=1;i<=n;i++) if (sa[i]>j) y[++cnt]=sa[i]-j;
				for (i=1;i<=n;i++) ++s[x[i]];
				for (i=2;i<=m;i++) s[i]+=s[i-1];
				for (i=n;i;i--) sa[s[x[y[i]]]--]=y[i];
				y[sa[1]]=cnt=1;
				memset(s+1,0,m*sizeof s[0]);
				for (i=2;i<=n;i++) if (x[sa[i]]==x[sa[i-1]]&&sa[i]<=n-j&&sa[i-1]<=n-j&&x[sa[i]+j]==x[sa[i-1]+j]) y[sa[i]]=cnt; else y[sa[i]]=++cnt;
				memcpy(x,y,sizeof(y));
				if ((m=cnt)==n) break;
			}
			for (i=1;i<=n;i++) rk[sa[i]]=i;
			j=0;
			for (i=1;i<=n;i++) if (x[i]>1)
			{
				cnt=sa[x[i]-1];
				while (i+j<=n&&cnt+j<=n&&a[i+j]==a[cnt+j]) ++j;
				h[x[i]]=j;
				if (j) --j;
			}
			st=vector<vector<int>>(lg[n]+1,vector<int>(n+1));
			for (i=1;i<=n;i++) st[0][i]=h[i];
			for (j=1;j<=lg[n];j++) for (i=1,k=n-(1<<j)+1;i<=k;i++) st[j][i]=min(st[j-1][i],st[j-1][i+(1<<j-1)]);
		}
	};
}
#define str SA::Q
```

<div style="page-break-after:always"></div>

### 图论

#### 最小密度环

$O(nm)$。

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=3e3+5,M=1e4+5;
const double inf=1e18;
int u[M],v[M];
double f[N][N],w[M];
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	cout<<setiosflags(ios::fixed)<<setprecision(8);
	int n,m,i,j;
	cin>>n>>m;
	for (i=1;i<=m;i++) cin>>u[i]>>v[i]>>w[i];
	++n;
	for (i=1;i<=n;i++)
	{
		fill_n(f[i]+1,n,inf);
		for (j=1;j<=m;j++) f[i][v[j]]=min(f[i][v[j]],f[i-1][u[j]]+w[j]);
	}
	double ans=inf;
	for (i=1;i<n;i++) if (f[n][i]!=inf)
	{
		double r=-inf;
		for (j=1;j<n;j++) r=max(r,(f[n][i]-f[j][i])/(n-j));
		ans=min(ans,r);
	}
	cout<<ans<<endl;
}
```

#### 全源最短路与判负环

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int,int> pa;
typedef tuple<int,int,int> tp;
const int N=152;
const ll inf=5e8;
ll dis[N][N],d[N][N];
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	while (1)
	{
		int n,m,q,i,j,k;
		cin>>n>>m>>q;
		if (tp(n,m,q)==tp(0,0,0)) return 0;
		for (i=0;i<n;i++) fill_n(dis[i],n,inf*inf);
		for (i=0;i<n;i++) dis[i][i]=0;
		while (m--)
		{
			int u,v,w;
			cin>>u>>v>>w;
			dis[u][v]=min(dis[u][v],(ll)w);
		}
		for (k=0;k<n;k++) for (i=0;i<n;i++) for (j=0;j<n;j++) dis[i][j]=max(min(dis[i][j],dis[i][k]+dis[k][j]),-inf*2);
		for (i=0;i<n;i++) copy_n(dis[i],n,d[i]);
		for (k=0;k<n;k++) for (i=0;i<n;i++) for (j=0;j<n;j++) dis[i][j]=max(min(dis[i][j],dis[i][k]+dis[k][j]),-inf*2);
		while (q--)
		{
			int u,v;
			cin>>u>>v;
			if (d[u][v]>inf) cout<<"Impossible\n"; else if (dis[u][v]!=d[u][v]||d[u][v]<-inf) cout<<"-Infinity\n"; else cout<<d[u][v]<<'\n';
		}
		cout<<'\n';
	}
}
```

#### 三元环计数

$O(m\sqrt m)$，$O(n+m)$。

```cpp
typedef pair<int,int> pa;
typedef tuple<int,int,int> tu;
vector<tu> get_triangles(int n,vector<pa> eg)//[0,n]
{
	++n;
	vector<tu> r;
	vector<int> e[n];
	int d[n],ed[n];
	fill_n(d,n,0);fill_n(ed,n,0);
	for (auto [x,y]:eg) ++d[y];
	for (auto [x,y]:eg)
	{
		if (pa{d[y],y}<pa{d[x],x}) swap(x,y);
		e[x].push_back(y);
	}
	for (int u=0;u<n;u++)
	{
		for (int v:e[u]) ed[v]=1;
		for (int v:e[u]) for (int w:e[v]) if (ed[w]) r.push_back({u,v,w});
		for (int v:e[u]) ed[v]=0;
	}
	return r;
}
```

#### Johnson 全源带负权最短路

$O(nm\log m)$，$O(n+m)$。

```cpp
for (int u=1;u<=n;u++) for (auto &[v,w]:e[u]) w+=dis[u]-dis[v];
```

#### 弦图判定

注意 add 是双向的。

```cpp
		if (n==0) return m;
		while (m--)
		{
			read(x);read(y);
			add(x,y);
		}
		num[0]=n+1;
		for (i=1;i<=n;i++) bel[0].push_back(i);
		for (i=n;i;i--)
		{
			while (1)
			{
				if (bel[zd].size()==0) --zd;
				if (gs[bel[zd][bel[zd].size()-1]]!=zd) bel[zd].pop_back(); else break;
			}
			num[dl[++wei]=x=bel[zd][bel[zd].size()-1]]=i;
			bel[zd].pop_back();
			if (bel[zd].size()==0) --zd;
			for (j=fir[x];j;j=nxt[j]) if (!num[lj[j]])
			{
				if (++gs[lj[j]]>zd) ++zd;
				bel[gs[lj[j]]].push_back(lj[j]);
			}
		}
		while (tou<=wei)
		{
			x=dl[tou++];y=0;
			for (i=fir[x];i;i=nxt[i]) if ((num[lj[i]]>num[x])&&(num[lj[i]]<num[y])) y=lj[i];
			for (i=fir[x];i;i=nxt[i]) if ((num[lj[i]]>num[x])&&(lj[i]!=y)) ed[st[++tp]=lj[i]]=1;
			for (i=fir[y];i;i=nxt[i]) if (ed[lj[i]]) ed[lj[i]]=0;
			while (tp) if (ed[st[tp--]])
			{
				ed[st[tp+1]]=0;suc=1;
			}
			if (suc) break;
		}
		if (suc) printf("Imperfect\n"); else printf("Perfect\n");
```

#### 弦图染色

注意 add 是双向的

```cpp
	read(n);read(m);
	while (m--)
	{
		read(x);read(y);
		add(x,y);
	}
	for (i=1;i<=n;i++) sz[0].push_back(i);
	for (i=n;i;i--)
	{
		while (1)
		{
			if (sz[zd].size()==0) --zd;
			if (cs[sz[zd][sz[zd].size()-1]]!=zd) sz[zd].pop_back(); else break;
		}
		ed[dl[++wei]=x=sz[zd][sz[zd].size()-1]]=1;
		sz[zd].pop_back();
		if (sz[zd].size()==0) --zd;
		for (j=fir[x];j;j=nxt[j]) if (!ed[lj[j]])
		{
			sz[++cs[lj[j]]].push_back(lj[j]);
			if (cs[lj[j]]>zd) ++zd;
		}
	}
	memset(ed,false,sizeof(ed));ed[0]=1;
	for (i=1;i<=n;i++)
	{
		zd=1;
		x=dl[i];
		for (j=fir[x];j;j=nxt[j]) if (!ed[col[lj[j]]])
		{
			ed[st[++tp]=col[lj[j]]]=1;
			while (ed[zd]) ++zd;
		}
		col[x]=zd;
		if (zd>ans) ans=zd;
		while (tp) ed[st[tp--]]=0;
	}
```

#### 二分图与网络流建图

以下约定，若为二分图则 $n,m$ 表示两侧点数，否则仅 $n$ 表示全图点数。

##### 二分图最小点集覆盖

$ans=\text{maxmatch}$，方案如下。

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=5e3+2;
vector<int> e[N];
int ed[N],lk[N],kl[N],flg[N],now;
bool dfs(int u)
{
	for (int v:e[u]) if (ed[v]!=now)
	{
		ed[v]=now;
		if (!lk[v]||dfs(lk[v])) return lk[v]=u;
	}
	return 0;
}
void dfs2(int u)
{
	for (int v:e[u]) if (!flg[v]) flg[v]=1,dfs2(lk[v]);
}
int main()
{
	int n,m,i,r=0;
	cin>>n>>m;
	while (m--)
	{
		int u,v;
		cin>>u>>v;
		e[u].push_back(v);
	}
	for (i=1;i<=n;i++) dfs(now=i);
	for (i=1;i<=n;i++) kl[lk[i]]=i;
	for (i=1;i<=n;i++) if (!kl[i]) dfs2(i);
	vector<int> A[2];
	for (i=1;i<=n;i++) if (lk[i])
	{
		if (flg[i]) A[1].push_back(i); else A[0].push_back(lk[i]);
	}
	for (int j=0;j<2;j++)
	{
		cout<<A[j].size();
		for (int x:A[j]) cout<<' '<<x;cout<<'\n';
	}
}
```

##### 二分图最大独立集

$ans=n+m-\text{maxmatch}$，方案是最小点集覆盖的补集。

##### 有向无环图最小不相交链覆盖

$ans=n-\text{maxmatch}$，其中二分图建图方法是拆入点和出点（实现时直接跑一次二分图就行，不用额外处理），注意**不**需要传递闭包。方案如下。

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=152;
vector<int> e[N];
int lk[N],kl[N],ed[N],now;
bool dfs(int u)
{
	for (int v:e[u]) if (ed[v]!=now)
	{
		ed[v]=now;
		if (!lk[v]||dfs(lk[v])) return lk[v]=u;
	}
	return 0;
}
int main()
{
	int n,m,i;
	ios::sync_with_stdio(0);cin.tie(0);
	cin>>n>>m;
	while (m--)
	{
		int u,v;
		cin>>u>>v;
		e[u].push_back(v);
	}
	int r=0;
	for (i=1;i<=n;i++) r+=dfs(now=i);
	for (i=1;i<=n;i++) kl[lk[i]]=i;
	for (i=1;i<=n;i++) if (ed[i]!=-1)
	{
		vector<int> ans;
		int u=i;
		while (u)
		{
			ed[u]=-1;
			ans.push_back(u);
			u=kl[u];
		}
		for (int j=0;j<ans.size();j++) cout<<ans[j]<<" \n"[j+1==ans.size()];
	}
	cout<<n-r<<endl;
}
```

##### 有向无环图最大互不可达集

$ans=n-\text{maxmatch}$，其中二分图建图方法是拆入点和出点（实现时直接跑一次二分图就行，不用额外处理），注意**需要**传递闭包。方案？

##### 最大权闭合子图

若 $v_i>0$，$s\to i$ 流量 $v_i$；若 $v_i<0$，$i\to t$ 流量 $-v_i$。若原图 $u\to v$ 可花费 $w$ 代价违抗，流量 $w$，否则 $+\infin$ 。答案为 $\sum\limits_{v_i>0} v_i-\text{maxflow}$。方案？

#### 二分图匹配（时间戳写法）

```cpp
bool dfs(int u)
{
	for (int v:e[u]) if (ed[v]!=now)
	{
		ed[v]=now;
		if (!lk[v]||dfs(lk[v])) return lk[v]=u;
	}
	return 0;
}
```

#### 二分图最大权匹配

```cpp
namespace KM
{
	const int N=405;//点数
	typedef long long ll;//答案范围
	const ll inf=1e16;
	int lk[N],kl[N],pre[N],q[N],n,h,t;
	ll sl[N],e[N][N],lx[N],ly[N];
	bool edx[N],edy[N];
	bool ck(int v)
	{
		if (edy[v]=1,kl[v]) return edx[q[++t]=kl[v]]=1;
		while (v) swap(v,lk[kl[v]=pre[v]]);
		return 0;
	}
	void bfs(int u)
	{
		fill_n(sl+1,n,inf);
		memset(edx+1,0,n*sizeof edx[0]);
		memset(edy+1,0,n*sizeof edy[0]);
		q[h=t=1]=u;edx[u]=1;
		while (1)
		{
			while (h<=t)
			{
				int u=q[h++],v;
				ll d;
				for (v=1;v<=n;v++) if (!edy[v]&&sl[v]>=(d=lx[u]+ly[v]-e[u][v])) if (pre[v]=u,d) sl[v]=d; else if (!ck(v)) return;
			}
			int i;
			ll m=inf;
			for (i=1;i<=n;i++) if (!edy[i]) m=min(m,sl[i]);
			for (i=1;i<=n;i++)
			{
				if (edx[i]) lx[i]-=m;
				if (edy[i]) ly[i]+=m; else sl[i]-=m;
			}
			for (i=1;i<=n;i++) if (!edy[i]&&!sl[i]&&!ck(i)) return;
		}
	}
	template<typename TT> ll max_weighted_match(int N,const vector<tuple<int,int,TT>> &edges)//lk[[1,n]]->[1,n]
	{
		int i,j;n=N;
		memset(lk+1,0,n*sizeof lk[0]);
		memset(kl+1,0,n*sizeof kl[0]);
		memset(ly+1,0,n*sizeof ly[0]);
		for (i=1;i<=n;i++) fill_n(e[i]+1,n,0);//若不需保证匹配边最多，置 0 即可，否则 -inf/N
		for (auto [u,v,w]:edges) e[u][v]=max(e[u][v],(ll)w);
		for (i=1;i<=n;i++) lx[i]=*max_element(e[i]+1,e[i]+n+1);
		for (i=1;i<=n;i++) bfs(i);
		ll r=0;
		for (i=1;i<=n;i++) r+=e[i][lk[i]];
		return r;
	}
}
using KM::max_weighted_match,KM::lk,KM::kl,KM::e;
```

#### 一般图最大匹配

```cpp
namespace blossom_tree
{
	const int N=1005;
	vector<int> e[N];
	int lk[N],rt[N],f[N],dfn[N],typ[N],q[N];
	int id,h,t,n;
	int lca(int u,int v)
	{
		++id;
		while (1)
		{
			if (u)
			{
				if (dfn[u]==id) return u;
				dfn[u]=id;u=rt[f[lk[u]]];
			}
			swap(u,v);
		}
	}
	void blm(int u,int v,int a)
	{
		while (rt[u]!=a)
		{
			f[u]=v;
			v=lk[u];
			if (typ[v]==1) typ[q[++t]=v]=0;
			rt[u]=rt[v]=a;
			u=f[v];
		}
	}
	void aug(int u)
	{
		while (u)
		{
			int v=lk[f[u]];
			lk[lk[u]=f[u]]=u;
			u=v;
		}
	}
	void bfs(int root)
	{
		memset(typ+1,-1,n*sizeof typ[0]);
		iota(rt+1,rt+n+1,1);
		typ[q[h=t=1]=root]=0;
		while (h<=t)
		{
			int u=q[h++];
			for (int v:e[u])
			{
				if (typ[v]==-1)
				{
					typ[v]=1;f[v]=u;
					if (!lk[v]) return aug(v);
					typ[q[++t]=lk[v]]=0;
				} else if (!typ[v]&&rt[u]!=rt[v])
				{
					int a=lca(rt[u],rt[v]);
					blm(v,u,a);blm(u,v,a);
				}
			} 
		}
	}
	int max_general_match(int N,vector<pair<int,int>> edges)//[1,n]
	{
		n=N;id=0;
		memset(f+1,0,n*sizeof f[0]);
		memset(dfn+1,0,n*sizeof dfn[0]);
		memset(lk+1,0,n*sizeof lk[0]);
		int i;
		for (i=1;i<=n;i++) e[i].clear();
		mt19937 rnd(114);
		shuffle(all(edges),rnd);
		for (auto [u,v]:edges)
		{
			e[u].push_back(v),e[v].push_back(u);
			if (!(lk[u]||lk[v])) lk[u]=v,lk[v]=u;
		}
		int r=0;
		for (i=1;i<=n;i++) if (!lk[i]) bfs(i);
		for (i=1;i<=n;i++) r+=!!lk[i];
		return r/2;
	}
}
using blossom_tree::max_general_match,blossom_tree::lk;
```

#### 一般图最大权匹配

$n=400$：UOJ 600ms, Luogu 135ms

```cpp
#include<bits/stdc++.h>
using namespace std;
#define all(x) (x).begin(),(x).end()
namespace weighted_blossom_tree
{
	#define d(x) (lab[x.u]+lab[x.v]-e[x.u][x.v].w*2)
	const int N=403*2;//两倍点数
	typedef long long ll;//总和大小
	typedef int T;//权值大小
	//均不允许无符号
	const T inf=numeric_limits<int>::max()>>1;
	struct Q
	{
		int u,v;
		T w;
	} e[N][N];
	T lab[N];
	int n,m=0,id,h,t,lk[N],sl[N],st[N],f[N],b[N][N],s[N],ed[N],q[N];
	vector<int> p[N];
	void upd(int u,int v) {if (!sl[v]||d(e[u][v])<d(e[sl[v]][v])) sl[v]=u;}
	void ss(int v)
	{
		sl[v]=0;
		for (int u=1;u<=n;u++) if (e[u][v].w>0&&st[u]!=v&&!s[st[u]]) upd(u,v);
	}
	void ins(int u) {if (u<=n) q[++t]=u; else for (int v:p[u]) ins(v);}
	void mdf(int u,int w)
	{
		st[u]=w;
		if (u>n) for (int v:p[u]) mdf(v,w);
	}
	int gr(int u,int v)
	{
		if ((v=find(all(p[u]),v)-p[u].begin())&1)
		{
			reverse(1+all(p[u]));
			return (int)p[u].size()-v;
		}
		return v;
	}
	void stm(int u,int v)
	{
		lk[u]=e[u][v].v;
		if (u<=n) return;
		Q w=e[u][v];
		int x=b[u][w.u],y=gr(u,x),i;
		for (i=0;i<y;i++) stm(p[u][i],p[u][i^1]);
		stm(x,v);
		rotate(p[u].begin(),y+all(p[u]));
	}
	void aug(int u,int v)
	{
		int w=st[lk[u]];
		stm(u,v);
		if (!w) return;
		stm(w,st[f[w]]);
		aug(st[f[w]],w);
	}
	int lca(int u,int v)
	{
		for (++id;u|v;swap(u,v))
		{
			if (!u) continue;
			if (ed[u]==id) return u;
			ed[u]=id;//??????????v?? 这是原作者的注释，我也不知道是啥
			if (u=st[lk[u]]) u=st[f[u]];
		}
		return 0;
	}
	void add(int u,int a,int v)
	{
		int x=n+1,i,j;
		while (x<=m&&st[x]) ++x;
		if (x>m) ++m;
		lab[x]=s[x]=st[x]=0;lk[x]=lk[a];
		p[x].clear();p[x].push_back(a);
		for (i=u;i!=a;i=st[f[j]]) p[x].push_back(i),p[x].push_back(j=st[lk[i]]),ins(j);//复制，改一处
		reverse(1+all(p[x]));
		for (i=v;i!=a;i=st[f[j]]) p[x].push_back(i),p[x].push_back(j=st[lk[i]]),ins(j);
		mdf(x,x);
		for (i=1;i<=m;i++) e[x][i].w=e[i][x].w=0;
		memset(b[x]+1,0,n*sizeof b[0][0]);
		for (int u:p[x])
		{
			for (v=1;v<=m;v++) if (!e[x][v].w||d(e[u][v])<d(e[x][v])) e[x][v]=e[u][v],e[v][x]=e[v][u];
			for (v=1;v<=n;v++) if (b[u][v]) b[x][v]=u;
		}
		ss(x);
	}
	void ex(int u)  // s[u] == 1
	{
		for (int x:p[u]) mdf(x,x);
		int a=b[u][e[u][f[u]].u],r=gr(u,a),i;
		for (i=0;i<r;i+=2)
		{
			int x=p[u][i],y=p[u][i+1];
			f[x]=e[y][x].u;
			s[x]=1;s[y]=0;
			sl[x]=0;ss(y);
			ins(y);
		}
		s[a]=1;f[a]=f[u];
		for (i=r+1;i<p[u].size();i++) s[p[u][i]]=-1,ss(p[u][i]);
		st[u]=0;
	}
	bool on(const Q &e)
	{
		int u=st[e.u],v=st[e.v],a;
		if(s[v]==-1)
		{
			f[v]=e.u;s[v]=1;
			a=st[lk[v]];
			sl[v]=sl[a]=s[a]=0;
			ins(a);
		}
		else if(!s[v])
		{
			a=lca(u,v);
			if (!a) return aug(u,v),aug(v,u),1;
			else add(u,a,v);
		}
		return 0;
	}
	bool bfs()
	{
		memset(s+1,-1,m*sizeof s[0]);
		memset(sl+1,0,m*sizeof sl[0]);
		h=1;t=0;
		int i,j;
		for (i=1;i<=m;i++) if (st[i]==i&&!lk[i]) f[i]=s[i]=0,ins(i);
		if (h>t) return 0;
		while (1)
		{
			while (h<=t)
			{
				int u=q[h++],v;
				if (s[st[u]]!=1) for (v=1; v<=n;v++) if (e[u][v].w>0&&st[u]!=st[v])
				{
					if (d(e[u][v])) upd(u,st[v]); else if (on(e[u][v])) return 1;
				}
			}
			T x=inf;
			for (i=n+1;i<=m;i++) if (st[i]==i&&s[i]==1) x=min(x,lab[i]>>1);
			for (i=1;i<=m;i++) if (st[i]==i&&sl[i]&&s[i]!=1) x=min(x,d(e[sl[i]][i])>>s[i]+1);
			for (i=1;i<=n;i++) if (~s[st[i]]) if ((lab[i]+=(s[st[i]]*2-1)*x)<=0) return 0;
			for (i=n+1;i<=m;i++) if (st[i]==i&&~s[st[i]]) lab[i]+=(2-s[st[i]]*4)*x;
			h=1;t=0;
			for (i=1;i<=m;i++) if (st[i]==i&&sl[i]&&st[sl[i]]!=i&&!d(e[sl[i]][i])&&on(e[sl[i]][i])) return 1;
			for (i=n+1;i<=m;i++) if (st[i]==i&&s[i]==1&&!lab[i]) ex(i);
		}
		return 0;
	}
	template<typename TT> ll max_weighted_general_match(int N,const vector<tuple<int,int,TT>> &edges)//[1,n]，返回权值
	{
		memset(ed+1,0,m*sizeof ed[0]);
		memset(lk+1,0,m*sizeof lk[0]);
		n=m=N;id=0;
		iota(st+1,st+n+1,1);
		int i,j;
		T wm=0;
		ll r=0;
		for (i=1;i<=n;i++) for (j=1;j<=n;j++) e[i][j]={i,j,0};
		for (auto [u,v,w]:edges) wm=max(wm,e[v][u].w=e[u][v].w=max(e[u][v].w,(T)w));
		for (i=1;i<=n;i++) p[i].clear();
		for (i=1;i<=n;i++) for (j=1;j<=n;j++) b[i][j]=i*(i==j);
		fill_n(lab+1,n,wm);
		while (bfs());
		for (i=1;i<=n;i++) if (lk[i]) r+=e[i][lk[i]].w;
		return r/2;
	}
	#undef d
}
using weighted_blossom_tree::max_weighted_general_match,weighted_blossom_tree::lk;
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	int n,m;
	cin>>n>>m;
	vector<tuple<int,int,long long>> edges(m);
	for (auto &[u,v,w]:edges) cin>>u>>v>>w;
	cout<<max_weighted_general_match(n,edges)<<'\n';
	for (int i=1;i<=n;i++) cout<<lk[i]<<" \n"[i==n];
}
```



#### 最大流

```cpp
namespace flow
{
	typedef int ui;//single flow
	typedef long long ll;//total flow
	const int N=4e5+2;//number of points
	const ll inf=numeric_limits<ll>::max();//maximum
	struct Q
	{
		int v;
		ui w;
		int id;
	};
	vector<Q> e[N];
	int fc[N],q[N];
	int n,s,t;
	int bfs()
	{
		fill_n(fc,n,0);
		int p1=0,p2=0,u;
		fc[s]=1;q[0]=s;
		while (p1<=p2)
		{
			int u=q[p1++];
			for (auto [v,w,id]:e[u]) if (w&&!fc[v]) fc[q[++p2]=v]=fc[u]+1;
		}
		return fc[t];
	}
	ll dfs(int u,ll maxf)
	{
		if (u==t) return maxf;
		ll j=0,k;
		for (auto &[v,w,id]:e[u]) if (w&&fc[v]==fc[u]+1&&(k=dfs(v,min(maxf-j,(ll)w))))
		{
			j+=k;
			w-=k;
			e[v][id].w+=k;
			if (j==maxf) return j;
		}
		fc[u]=0;
		return j;
	}
	ll maxflow(int N,const vector<tuple<int,int,ui>> &edges,int S,int T)//[0,n]
	{
		s=S;t=T;n=N;++n;
		for (int i=0;i<n;i++) e[i].clear();
		for (auto [u,v,w]:edges) if (u!=v)
		{
			e[u].push_back({v,w,(int)e[v].size()});
			e[v].push_back({u,0,(int)e[u].size()-1});
		}
		ll r=0;
		while (bfs()) r+=dfs(s,inf);
		return r;
	}
}
using flow::maxflow,flow::fc,flow::e;
namespace match
{
	const int N=4e5+2;//number of points
	int lk[N];
	int maxmatch(int n,int m,const vector<pair<int,int>> &edges)//lk[[0,n]]->[0,m]
	{
		++n;++m;
		int s=n+m,t=n+m+1,i;
		vector<tuple<int,int,int>> eg;
		eg.reserve(n+m+edges.size());
		for (i=0;i<n;i++) eg.push_back({s,i,1});
		for (i=0;i<m;i++) eg.push_back({i+n,t,1});
		for (auto [u,v]:edges) eg.push_back({u,v+n,1});
		int r=maxflow(t,eg,s,t);
		fill_n(lk,n,-1);
		for (i=0;i<n;i++) for (auto [v,w,id]:e[i]) if (v<s&&!w) {lk[i]=v-n;break;}
		return r;
	}
}
using match::maxmatch,match::lk;
namespace costflow
{
	const int N=1e5+2;//number of point
	typedef int wT;
	typedef long long cT;
	const cT inf=numeric_limits<cT>::max();
	struct Q
	{
		int v;
		wT w;
		cT c;
		int id;
	};
	vector<Q> e[N];
	cT dis[N];
	int pre[N],pid[N],ipd[N];
	bool ed[N];
	int n,s,t;
	pair<wT,cT> spfa()
	{
		queue<int> q;
		fill_n(dis,n,inf);
		q.push(s);dis[s]=0;
		while (q.size())
		{
			int u=q.front();q.pop();ed[u]=0;
			for (auto [v,w,c,id]:e[u]) if (w&&dis[v]>dis[u]+c)
			{
				dis[v]=dis[u]+c;
				pre[v]=u;
				pid[v]=e[v][id].id;
				ipd[v]=id;
				if (!ed[v]) q.push(v),ed[v]=1;
			}
		}
		if (dis[t]==inf) return {0,0};
		wT mw=numeric_limits<wT>::max();
		for (int i=t;i!=s;i=pre[i]) mw=min(mw,e[pre[i]][pid[i]].w);
		for (int i=t;i!=s;i=pre[i]) e[pre[i]][pid[i]].w-=mw,e[i][ipd[i]].w+=mw;
		return {mw,(cT)mw*dis[t]};
	}
	pair<wT,cT> mincostflow(int N,const vector<tuple<int,int,wT,cT>> &edges,int S,int T)//[0,n]
	{
		n=++N;s=S;t=T;
		for (int i=0;i<n;i++) e[i].clear();
		for (auto [u,v,w,c]:edges) if (u!=v)
		{
			e[u].push_back({v,w,c,(int)e[v].size()});
			e[v].push_back({u,0,-c,(int)e[u].size()-1});
		}
		pair<wT,cT> r{0,0},rr;
		while ((rr=spfa()).first) r={r.first+rr.first,r.second+rr.second};
		return r;
	}
}
using costflow::mincostflow,costflow::e;
```

#### 费用流（SPFA）

```cpp
bool dfs()
{
	memset(jl,-0x3f,sizeof(jl));
	jl[dl[tou=wei=1]=0]=0;
	while (tou<=wei)
	{
		ed[x=dl[tou++]]=0;
		for (i=fir[x];i;i=nxt[i]) if ((lj[i][1])&&(jl[lj[i][0]]<jl[x]+lj[i][2]))
		{
			jl[lj[i][0]]=jl[x]+lj[i][2];
			qq[lj[i][0]]=x;
			dy[lj[i][0]]=i;
			if (!ed[lj[i][0]]) ed[dl[++wei]=lj[i][0]]=1;
		}
	}
	zg=m;
	if (jl[t]==jl[t+1]) return 0;
	for (i=t;i;i=qq[i]) zg=min(zg,lj[dy[i]][1]);
	for (i=t;i;i=qq[i])
	{
		lj[dy[i]][1]-=zg;
		ans+=zg*lj[dy[i]][2];
		if (dy[i]&1) lj[dy[i]+1][1]+=zg; else lj[dy[i]-1][1]+=zg;
	}
	return 1;
}
while (dfs());
```

#### 费用流（Dijkstra）

```cpp
priority_queue<pa,vector<pa>,greater<pa> > heap;
const int N=5e3+2,M=1e5+2;
pa ans;
int lj[M][3],nxt[M],fir[N],dis[N],h[N],pre[N],fl[N];
int n,m,s,t,bs,x,y,z,w,ans1,ans2;
bool ed[N];
void add(const int u,const int v,const int x,const int y)
{
	lj[++bs][0]=v;
	lj[bs][1]=x;
	lj[bs][2]=y;
	nxt[bs]=fir[u];
	fir[u]=bs;
	lj[++bs][0]=u;
	lj[bs][1]=0;
	lj[bs][2]=-y;
	nxt[bs]=fir[v];
	fir[v]=bs;
}
void spfa()//本题中用dijkstra代替
{
	int x,i,j;
	memset(h,0x3f,sizeof(h));h[s]=0;
	heap.push(make_pair(0,s));
	while (!heap.empty())
	{
		ed[x=heap.top().second]=1;heap.pop();
		for (i=fir[x];i;i=nxt[i]) if ((lj[i][1])&&(h[lj[i][0]]>h[x]+lj[i][2]))
			heap.push(make_pair(h[lj[i][0]]=h[x]+lj[i][2],lj[i][0]));
		while ((!heap.empty())&&(ed[heap.top().second])) heap.pop();
	}
	for (i=1;i<=n;i++) for (j=fir[i];j;j=nxt[j]) lj[j][2]+=h[i]-h[lj[j][0]];
	memset(ed,0,sizeof(ed));
}
pa dijkstra()
{
	int i,j,x,tf=1e9;
	memset(dis,0x3f,sizeof(dis));memset(pre,0,sizeof(pre));dis[s]=0;heap.push(make_pair(0,s));
	while (!heap.empty())
	{
		ed[x=heap.top().second]=1;heap.pop();
		for (i=fir[x];i;i=nxt[i]) if ((lj[i][1])&&(dis[lj[i][0]]>dis[x]+lj[i][2]))
			heap.push(make_pair(dis[lj[i][0]]=dis[pre[lj[i][0]]=x]+lj[i][2],lj[i][0])),fl[lj[i][0]]=i;
		while ((!heap.empty())&&(ed[heap.top().second])) heap.pop();
	}
	if (dis[t]==dis[t+1]) return make_pair(0,0);
	for (i=t;i!=s;i=pre[i]) tf=min(tf,lj[fl[i]][1]);
	for (i=t;i!=s;i=pre[i]) lj[fl[i]][1]-=tf,lj[fl[i]^1][1]+=tf;
	for (i=1;i<=n;i++) for (j=fir[i];j;j=nxt[j]) lj[j][2]+=dis[i]-dis[lj[j][0]];
	h[t]+=dis[t];memset(ed,0,sizeof(ed));
	return make_pair(tf,tf*h[t]);
}
signed main()
{
	while (!heap.empty()) heap.pop();
	read(n);read(m);read(s);read(t);bs=1;
	while (m--)
	{
		read(x);read(y);read(z);read(w);
		add(x,y,z,w);
	}
	spfa();
	while ((ans=dijkstra()).first) ans1+=ans.first,ans2+=ans.second;
	printf("%d %d",ans1,ans2);
}
```



#### 假花树

```cpp
vector<int> lj[N];
int lk[N],ed[N];
int n,m,cnt,i,t,x,y,ans,la;
bool dfs(int x)
{
	ed[x]=cnt;int v;
	random_shuffle(lj[x].begin(),lj[x].end());
	for (auto u:lj[x]) if (ed[v=lk[u]]!=cnt)
	{
		lk[v]=0,lk[u]=x,lk[x]=u;
		if (!v||dfs(v)) return 1;
		lk[v]=u,lk[u]=v,lk[x]=0;
	}
	return 0;
}
int main()
{
	srand(time(0));la=-1;
	read(n);read(m);
	while (m--) read(x),read(y),lj[x].push_back(y),lj[y].push_back(x);
	while (la!=ans)
	{
		memset(ed+1,0,n<<2);la=ans;
		for (i=1;i<=n;i++) if (!lk[i]) ans+=dfs(cnt=i);
	}
	printf("%d\n",ans);
	for (i=1;i<=n;i++) printf("%d ",lk[i]);
}
```

#### Stoer-Wagner 全局最小割

$O(n^3)$。可优化到 $O(nm\log n)$。

```cpp
namespace StoerWagner
{
	const int N=602;//点数
	typedef int T;//边权和
	T e[N][N],w[N];
	int ed[N],p[N],f[N];//f 仅输出方案用
	int getf(int u){return f[u]==u?u:f[u]=getf(f[u]);}
	template<typename TT> pair<T,vector<int>> mincut(int n,const vector<tuple<int,int,TT>> &edges)//[1,n]，返回某一半点集
	{
		vector<int> ans;ans.reserve(n);
		int i,j,m;
		T r;
		r=numeric_limits<T>::max();
		for (i=1;i<=n;i++) memset(e[i]+1,0,n*sizeof e[0][0]);
		for (auto [u,v,w]:edges) e[u][v]+=w,e[v][u]+=w;
		fill_n(ed+1,n,0);
		iota(f+1,f+n+1,1);
		for (m=n;m>1;m--)
		{
			fill_n(w+1,n,0);
			for (i=1;i<=n;i++) ed[i]&=2;
			for (i=1;i<=m;i++)
			{
				int x=0;
				for (j=1;j<=n;j++) if (!ed[j]) break;x=j;
				for (j++;j<=n;j++) if (!ed[j]*w[j]>w[x]) x=j;
				ed[p[i]=x]=1;
				for (j=1;j<=n;j++) w[j]+=!ed[j]*e[x][j];
			}
			int s=p[m-1],t=p[m];
			if (r>w[t])
			{
				r=w[t];ans.clear();
				for (i=1;i<=n;i++) if (getf(i)==getf(t)) ans.push_back(i);
			}
			for (i=1;i<=n;i++) e[i][s]=e[s][i]+=e[t][i];
			ed[t]=2;
			f[getf(s)]=getf(t);
		}
		return {r,ans};
	}
}
```

#### 点双

$O(n+m)$，$O(n+m)$。

```cpp
int lj[M],nxt[M],fir[N],dfn[N],low[N];
int n,m,i,x,y,c,bs;
bool ct[N];
void add()
{
	lj[++bs]=y;
	nxt[bs]=fir[x];
	fir[x]=bs;
	lj[++bs]=x;
	nxt[bs]=fir[y];
	fir[y]=bs;
}
void dfs(int x,int rt)
{
	dfn[x]=low[x]=++bs;
	int i,hs=0;
	for (i=fir[x];i;i=nxt[i]) if (dfn[lj[i]])
	{
		low[x]=min(low[x],dfn[lj[i]]);
	}
	else
	{
		dfs(lj[i],0);++hs;
		low[x]=min(low[x],low[lj[i]]);
		if ((dfn[x]==low[lj[i]])&&(!rt)) ct[x]=1;
	}
	if ((rt)&&(hs>1)) ct[x]=1;
}
int main()
{
	read(n);read(m);
	while (m--) {read(x);read(y);add();}bs=0;
	for (i=1;i<=n;i++) if (!dfn[i]) dfs(i,1);
	bs=0;
	for (i=1;i<=n;i++) bs+=ct[i];
	printf("%d\n",bs);
	for (i=1;i<=n;i++) if (ct[i]) printf("%d ",i);
}
```

#### 边双

$O(n+m)$，$O(n+m)$。

```cpp
void dfs2(int x,int fa)
{
	dfn[x]=low[x]=++id;
	for (int i=1;i<=n;i++) if (mp[x][i]&&i!=fa)
	{
		if (!dfn[i])
		{
			dfs2(i,x);low[x]=min(low[x],low[i]);
			if (dfn[x]<low[i]) ct[x][i]=ct[i][x]=1;
		} else low[x]=min(low[x],dfn[i]);
	}
}
```

#### 输出负环

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=34;
struct Q
{
	int v,w,c;
	Q(){}
	Q(int x,int y,int z):v(x),w(y),c(z){}
};
vector<Q> lj[N];
int dis[N],cnt[N],pt[N],S;
Q pre[N],st[N];
int n,m,ans,tp;
bool ed[N];
int main()
{
	freopen("arbitrage.in","r",stdin);
	freopen("arbitrage.out","w",stdout);
	ios::sync_with_stdio(0);cin.tie(0);
	cin>>n>>m;
	while (m--)
	{
		int x,y,z,w;
		cin>>x>>y>>z>>w;
		lj[x].emplace_back(y,w,z);
		lj[y].emplace_back(x,0,-z);
	}
	for (int i=1;i<=n;i++) lj[0].emplace_back(i,1,0);
	while (1)
	{
		memset(dis,-0x3f,sizeof dis);dis[0]=0;
		for (int i=0;i<=n;i++) ed[i]=cnt[i]=0;S=-1;
		queue<int> q;q.push(0);
		while (!q.empty())
		{
			int u=q.front();q.pop();ed[u]=0;
			for (auto &[v,w,c]:lj[u]) if (w&&dis[v]<dis[u]+c)
			{
				dis[v]=dis[u]+c;pre[v]=Q(u,w,c);
				if (!ed[v])
				{
					if (++cnt[v]>n+1) {S=v;goto aa;}
					ed[v]=1;q.push(v);
				}
			}
		}
		aa:;
		if (S==-1) break;
		{
			static bool ed[N];
			memset(ed,0,sizeof ed);
			while (!ed[S]) ed[S]=1,S=pre[S].v;
		}
		st[tp=1]=pre[S];pt[1]=S;
		int x=pre[S].v;
		while (x!=S)
		{
			st[++tp]=pre[x];pt[tp]=x;
			x=pre[x].v;
			assert(tp<=n+5);
		}
		int fl=1e9;
		for (int j=1;j<=tp;j++) fl=min(fl,st[j].w);
		assert(fl);
		for (int j=1;j<=tp;j++)
		{
			ans+=fl*st[j].c;
			int nn=0;
			for (auto &[v,w,c]:lj[st[j].v]) if (v==pt[j]&&st[j].c==c&&st[j].w==w) {++nn;w-=fl;break;}
			for (auto &[v,w,c]:lj[pt[j]]) if (v==st[j].v&&st[j].c+c==0) {++nn;w+=fl;break;}assert(nn==2);
		}
	}
	cout<<ans<<endl;
}
```



#### DAG 删点最长路

$O((n+m)\log n)$，$O(n+m)$。

```cpp
priority_queue<int> hp1,hp2,del1,del2;
int lj[M],nxt[M],fir[N],flj[M],fnxt[M],ffir[N],dl[N],rd[N],cd[N],dis1[N],dis2[N];
int dtp;
char c[M*15+1];
int main()
{
	int n,m,i,j,x,y,tou,wei,zd=0,ans=M,cur,pos=0;
	scanf("%d%d",&n,&m);
	fread(c+1,1,m*15,stdin);
	for (i=1;i<=m;i++)
	{
		read(x);read(y);++cd[x];
		lj[i]=y;nxt[i]=fir[x];fir[x]=i;++rd[y];
		flj[i]=x;fnxt[i]=ffir[y];ffir[y]=i;
	}
	tou=1;wei=0;
	for (i=1;i<=n;i++) if (!cd[i]) dl[++wei]=i;
	while (tou<=wei) for (i=ffir[x=dl[tou++]];i;i=fnxt[i]) 
	{
		dis2[flj[i]]=max(dis2[flj[i]],dis2[x]+1);
		if (--cd[flj[i]]==0) dl[++wei]=flj[i];
	}
	tou=1;wei=0;
	for (i=1;i<=n;i++) if (!rd[i]) dl[++wei]=i;
	while (tou<=wei) for (i=fir[x=dl[tou++]];i;i=nxt[i]) 
	{
		dis1[lj[i]]=max(dis1[lj[i]],dis1[x]+1);
		if (--rd[lj[i]]==0) dl[++wei]=lj[i];
	}
	for (i=1;i<=n;i++) hp1.push(dis2[i]);hp1.push(0);hp2.push(0);
	for (j=1;j<=wei;j++)
	{
		x=dl[j];
		if (dis2[x]==hp1.top())
		{
			hp1.pop();
			while ((!del1.empty())&&(del1.top()==hp1.top())) {hp1.pop();del1.pop();}
		} else del1.push(dis2[x]);
		for (i=ffir[x];i;i=fnxt[i]) del2.push(dis1[flj[i]]+dis2[x]+1);
		while ((!del2.empty())&&(del2.top()==hp2.top())) {hp2.pop();del2.pop();}
		cur=max(zd,max(hp1.top(),hp2.top()));
		if (cur<ans)
		{
			pos=dl[j];ans=cur;
		}
		zd=max(zd,dis1[x]);
		for (i=fir[x];i;i=nxt[i]) hp2.push(dis1[x]+dis2[lj[i]]+1);
		if (ans<=zd) break;
	}
	printf("%d %d",pos,ans);
}
```

#### （基环）树哈希

```cpp
#include <bits/stdc++.h>
using namespace std;
namespace tree_hash
{
	typedef unsigned int ui;
	typedef unsigned long long ll;
	const int N=1e6+2;
    const ui p1=2034452107,p2=2013074419,B=(1ll<<32)-1;
	mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
	ui bas1[N],bas2[N],lst;
	ui uni1,uni2;
	vector<int> e[N];
	vector<ll> rt;
	ll g[N];
	int siz[N],h[N],f[N],num[N*2];
	int n,m;
	void init()
	{
		uni1=rnd()%(p1/2)+p1/2;uni2=rnd()%(p2/2)+p2/2;
		lst=0;
	}
	void dfs1(int u)
	{
		siz[u]=1;
		int mx=0;
		for (auto &v:e[u]) if (v!=f[u])
		{
			f[v]=u;dfs1(v);siz[u]+=siz[v];
			mx=max(mx,siz[v]);
		}
		mx=max(mx,n-siz[u]);
		if (mx*2<=n) rt.push_back(u);
	}
	void dfs2(int u)
	{
		for (auto &v:e[u]) if (v!=f[u]) f[v]=u,dfs2(v),h[u]=max(h[u],h[v]);
		++h[u];
		int n=0;
		static ll a[N];
		for (auto &v:e[u]) if (v!=f[u]) a[n++]=g[v];
		sort(a,a+n);
		ll r1=0,r2=0;
		a[n++]=1ll<<32|1;
		for (int i=0;i<n;i++) r1=(r1*bas1[h[u]]+(a[i]>>32))%p1,r2=(r2*bas2[h[u]]+(a[i]&B))%p2;
		g[u]=r1<<32|r2;
	}
	void get_e(vector<pair<int,int>> &E)
	{
		int i;
		n=E.size()+1;m=0;
		for (auto &[u,v]:E) num[m++]=u,num[m++]=v;
		sort(num,num+m);m=unique(num,num+m)-num;
		for (i=0;i<m;i++) e[num[i]].clear();
		for (auto &[u,v]:E) e[u].push_back(v),e[v].push_back(u);
		while (lst<n) bas1[++lst]=rand()%(p1/2)+p1/2,bas2[lst]=rnd()%(p2/2)+p2/2;
	}
	ll rooted_tree_hash(int u)
	{
		if (n==1) return 1ll<<32|1;
		for (int i=0;i<m;i++) f[num[i]]=0,h[num[i]]=0;
		dfs2(u);
		return g[u];
	}
	ll t_h(vector<pair<int,int>> &E)
	{
		int i;
		get_e(E);
		for (i=0;i<m;i++) f[num[i]]=0;
		rt.clear();dfs1(1);
		ll r1=0,r2=0;
		for (auto &u:rt) u=rooted_tree_hash(u);
		sort(rt.begin(),rt.end());
		for (auto &u:rt) r1=(r1*uni1+(u>>32))%p1,r2=(r2*uni2+(u&B))%p2;
		return r1<<32|r2;
	}
}
using tree_hash::get_e;
using tree_hash::rooted_tree_hash;
using tree_hash::t_h;
typedef pair<int,int> pa;
typedef unsigned int ui;
typedef unsigned long long ull;
const ui mod1=2034452107,mod2=2013074419,B=(1ll<<32)-1;
ui b1,b2;
const int N=1e6+2;
vector<int> e[N];
int f[N];
vector<int> lp;
int getf(int u) {return f[u]==u?u:f[u]=getf(f[u]);}
void dfs1(int u)
{
	for (auto &v:e[u]) if (v!=f[u]) f[v]=u,dfs1(v);
}
bool ed[N];
void dfs2(int u,vector<pa> &E)
{
	for (auto &v:e[u]) if (!ed[v]) ed[v]=1,E.emplace_back(u,v),dfs2(v,E);
}
void min_order(ull *a,int n)
{
	int i,j,k;
	ull x,y;
	i=k=0;j=1;
	while ((i<n)&&(j<n)&&(k<n))
	{
		x=a[(i+k)%n];y=a[(j+k)%n];
		if (x==y) ++k; else
		{
			if (x>y) i+=k+1;  else j+=k+1;
			if (i==j) ++j;
			k=0;
		}
	}
	if (j>i) j=i;
	//[j,n)+[0,j)
	rotate(a,a+j,a+n);
}
int cal()
{
	int n,m,p1,p2;
	cin>>m;
	vector<pair<ull,ull>> a(m);
	for (auto &V:a)
	{
		int i;
		cin>>n;
		for (i=1;i<=n;i++) e[i].clear();
		iota(f+1,f+n+1,1);
		for (i=1;i<=n;i++)
		{
			int u,v;
			cin>>u>>v;
			if (getf(u)==getf(v)) {p1=u;p2=v;continue;}
			e[u].push_back(v);
			e[v].push_back(u);
			f[f[u]]=f[v];
		}
		memset(f+1,0,n*sizeof f[0]);
		dfs1(p1);
		static int st[N];
		memset(ed+1,0,n*sizeof ed[0]);
		int tp=1;st[1]=p2;
		while (p2!=p1) st[++tp]=p2=f[p2];
		for (i=1;i<=tp;i++) ed[st[i]]=1;
		vector<pa> E;
		static ull ans[N];
		E.reserve(n);
		for (i=1;i<=tp;i++)
		{
			dfs2(st[i],E);
			get_e(E);
			ans[i]=rooted_tree_hash(st[i]);
			E.clear();
		}
		min_order(ans+1,tp);
		ull r1=0,r2=0,r,rr;
		for (int i=1;i<=tp;i++) r1=(r1*b1+(ans[i]>>32))%mod1,r2=(r2*b2+(ans[i]&B))%mod2;
		r=r1<<32|r2;
		reverse(ans+1,ans+tp+1);
		min_order(ans+1,tp);r1=r2=0;
		for (int i=1;i<=tp;i++) r1=(r1*b1+(ans[i]>>32))%mod1,r2=(r2*b2+(ans[i]&B))%mod2;
		rr=r1<<32|r2;
		if (r>rr) swap(r,rr);
		V=make_pair(r,rr);
	}
	sort(a.begin(),a.end());
	return unique(a.begin(),a.end())-a.begin();
}
int main()
{
	b1=tree_hash::rnd()%(mod1/2)+mod1/2;
	b2=tree_hash::rnd()%(mod2/2)+mod2/2;
	tree_hash::init();
	ios::sync_with_stdio(0);cin.tie(0);
	int n,T;
	cin>>T;
	while (T--) cout<<cal()<<'\n';
}
```



#### 无向图最小环

$O(n^3)$，$O(n^2)$。

```cpp
int f[N][N],jl[N][N];
int n,m,c,ans=inf,i,j,k,x,y,z;
int main()
{
	read(n);read(m);
	memset(f,0x3f,sizeof(f));
	memset(jl,0x3f,sizeof(jl));
	while (m--)
	{
		read(x);read(y);read(z);
		jl[x][y]=jl[y][x]=f[x][y]=f[y][x]=min(f[y][x],z);
	}
	for (k=1;k<=n;k++)
	{
		for (i=1;i<k;i++) if (jl[k][i]!=jl[0][0]) for (j=1;j<i;j++)
			if (jl[k][j]!=jl[0][0]) ans=min(ans,jl[k][i]+jl[k][j]+f[i][j]);
		for (i=1;i<=n;i++) if (i!=k) for (j=1;j<=n;j++)
			if ((j!=i)&&(j!=k)) f[i][j]=min(f[i][j],f[i][k]+f[k][j]);
	}
	if (ans==inf) puts("No solution."); else printf("%d",ans);
}
```

#### 切比雪夫距离最小生成树

$O(n\log n)$，$O(n)$。

```cpp
const int N=3e5+2,M=N<<2;
struct P
{
	int u,v,w;
	P(int a=0,int b=0,int c=0):u(a),v(b),w(c){}
	bool operator<(const P &o) const {return w<o.w;}
};
struct Q
{
	int x,y,id;
	Q(int a=0,int b=0,int c=0):x(a),y(b),id(c){}
	bool operator<(const Q &o) const {return x!=o.x?x>o.x:y>o.y;}
};
ll ans;
P lb[M];
Q a[N],b[N];
int f[N],c[N];
int n,m,i,x,y;
struct bit
{
	int a[N],pos[N],n;
	void init(int &nn)
	{
		memset(a+1,0x7f,(n=nn)*sizeof a[0]);
		memset(pos+1,0,n*sizeof pos[0]);
	}
	void mdf(int x,const int y,const int z)
	{
		if (a[x]>y) a[x]=y,pos[x]=z;
		while (x-=x&-x) if (a[x]>y) a[x]=y,pos[x]=z;
	}
	int sum(int x)
	{
		int r=a[x],rr=pos[x];
		while ((x+=x&-x)<=n) if (a[x]<r) r=a[x],rr=pos[x];
		return rr;
	}
};
bit s;
void cal()
{
	int i,x,y;
	s.init(n);
	memcpy(b+1,a+1,sizeof(Q)*n);
	sort(a+1,a+n+1);
	for (i=1;i<=n;i++) c[i]=a[i].y-a[i].x;
	sort(c+1,c+n+1);
	for (i=1;i<=n;i++)
	{
		if (x=s.sum(y=lower_bound(c+1,c+n+1,a[i].y-a[i].x)-c))
			lb[++m]=P(a[x].id,a[i].id,a[x].x+a[x].y-a[i].x-a[i].y);//谨防 int 爆
		s.mdf(y,a[i].y+a[i].x,i);
	}
	memcpy(a+1,b+1,sizeof(Q)*n);
}
int getf(int x) {return f[x]==x?x:f[x]=getf(f[x]);}
int main()
{
	read(n);
	for (i=1;i<=n;i++) {read(a[f[i]=a[i].id=i].x);read(a[i].y);
		swap(a[i].x,a[i].y);a[i]=Q(a[i].x+a[i].y,a[i].x-a[i].y,i);}
	cal();for (i=1;i<=n;i++) swap(a[i].x,a[i].y);
	cal();for (i=1;i<=n;i++) a[i].y=-a[i].y;
	cal();for (i=1;i<=n;i++) swap(a[i].x,a[i].y);
	cal();sort(lb+1,lb+m+1);
	for (i=1;i<=m;i++) if ((x=getf(lb[i].u))!=(y=getf(lb[i].v))) f[x]=y,ans+=lb[i].w;
	printf("%lld\n",ans>>1);
}
```

#### 点分治

$O(n\log n)$，$O(n)$。

```cpp
void sol(int x)
{
	zd=n<<1;rt=x;
	dfs(x);
	ed[x=rt]=1;
	int i;
	tou=1;wei=0;
	for (i=fir[x];i;i=nxt[i]) if (!ed[lj[i][0]])
	{
		deep[lj[i][0]]=lj[i][1];
		qh(lj[i][0]);
		while (tou<=wei) ad(dl[tou++],1);
	}
	while (wei) ad(dl[wei--],0);
	int n=ksiz-siz[x];
	for (i=fir[x];i;i=nxt[i]) if (!ed[lj[i][0]])
	{
		//dfs(lj[i][0]);
		//ksiz=siz[lj[i][0]];
		ksiz=siz[lj[i][0]]>siz[x] ? n:siz[lj[i][0]];
		sol(lj[i][0]);
	}
}
```

#### prufer 与树的互相转化

$O(n)$，$O(n)$。

```cpp
vector<int> edges_to_prufer(const vector<pair<int,int>> &eg)//[1,n]，定根为 n
{
	int n=eg.size()+1,i,j,k;
	int fir[n+1],nxt[n*2+1],e[n*2+1];
	int rd[n+1],cnt=0;
	memset(rd,0,sizeof rd);memset(nxt,0,sizeof nxt);memset(fir,0,sizeof fir);
	for (auto [u,v]:eg)
	{
		e[++cnt]=v;nxt[cnt]=fir[u];fir[u]=cnt;++rd[v];
		e[++cnt]=u;nxt[cnt]=fir[v];fir[v]=cnt;++rd[u];
	}
	for (i=1;i<=n;i++) if (rd[i]==1) break;
	int u=i;
	vector<int> r;r.reserve(n-2);
	for (j=1;j<n-1;j++)
	{	
		for (k=fir[u],u=rd[u]=0;k;k=nxt[k]) if (rd[e[k]]) 
		{
			r.push_back(e[k]);
			if ((--rd[e[k]]==1)&&(e[k]<i)) u=e[k];
		}
		if (!u) { while (rd[i]!=1) ++i;u=i;}
	}
	return r;
}
vector<pair<int,int>> prufer_to_edges(const vector<int> &p)//[1,n]，定根为 n
{
	int n=p.size(),i,j,k;
	int m=n+3;
	int cs[m];memset(cs,0,sizeof cs);
	for (i=0;i<n;i++) ++cs[p[i]];
	i=0;
	while (cs[++i]);
	int u=i,v;
	vector<pair<int,int>> r;
	r.reserve(n-2);
	for (j=0;j<n;j++)
	{
		cs[u]=1e9;
		r.push_back({u,v=p[j]});
		if ((--cs[v]==0)&&(v<i)) u=v;
		if (v!=u) {while (cs[i]) ++i;u=i;}
	}
	r.push_back({u,n+2});
	return r;
}

```

#### LCT

$O(n\log n)$，$O(n)$。

makeroot 会变根，split 会把 $y$ 变根，findroot 会把根变根，link 会把 $x,y$ 变根（$y$ 是新的），cut 会把 $x,y$ 变根（$x$ 是新的），注意 swap 子节点可能要 pushup。

```cpp
template<int N,typename Q> struct LCT
{
	int f[N],c[N][2],siz[N],st[N];
	Q s[N],v[N];
	#ifdef Rev
	Q rs[N];
	#endif
	//heap g[N]; //虚子树
	bool lz[N];
	void init(int n)
	{
		++n;
		for (int i=0;i<n;i++)
		{
			f[i]=c[i][0]=c[i][1]=lz[i]=0;
			s[i]=v[i]=Q();
			#ifdef Rev
			rs[i]=Q();
			#endif
			siz[i]=!!i;
		}
	}
	void modify(int x,const Q &o)
	{
		makeroot(x);
		v[x]=o;
		pushup(x);
	}
	bool nroot(int x) const
	{
		return c[f[x]][0]==x||c[f[x]][1]==x;
	}
	void pushup(int x)
	{
		int lc=c[x][0],rc=c[x][1];
		s[x]=v[x];siz[x]=1;
		#ifdef Rev
		rs[x]=v[x];
		#endif
		if (lc)
		{
			s[x]=s[lc]+s[x];
			siz[x]+=siz[lc];
			#ifdef Rev
			rs[x]=rs[x]+rs[lc];
			#endif

		}
		if (rc)
		{
			s[x]=s[x]+s[rc];
			siz[x]+=siz[rc];
			#ifdef Rev
			rs[x]=rs[rc]+rs[x];
			#endif
		}
	}
	void swp(int x)
	{
		swap(c[x][0],c[x][1]);
		#ifdef Rev
		swap(s[x],rs[x]);
		#endif
		lz[x]^=1;
	}
	void pushdown(int x)
	{
		int lc=c[x][0],rc=c[x][1];
		if (lz[x])
		{
			if (lc) swp(lc);
			if (rc) swp(rc);
			lz[x]=0;
		}
	}
	void zigzag(int x)
	{
		int y=f[x],z=f[y],typ=(c[y][0]==x);
		if (nroot(y)) c[z][c[z][1]==y]=x;
		f[x]=z;f[y]=x;
		if (c[x][typ]) f[c[x][typ]]=y;
		c[y][typ^1]=c[x][typ];c[x][typ]=y;
		pushup(y);
	}
	void splay(int x)
	{
		int y,tp=0;
		st[tp=1]=y=x;
		while (nroot(y)) st[++tp]=y=f[y];
		while (tp) pushdown(st[tp--]);
		for (;nroot(x);zigzag(x)) if (!nroot(f[x])) continue; else zigzag((c[f[x]][0]==x)^(c[f[f[x]]][0]==f[x]) ? x:f[x]);
		pushup(x);
	}
	void access(int x)
	{
		for (int y=0;x;x=f[y=x])
		{
			splay(x);
			//g[x].ins(s[c[x][1]]);g[x].del(s[y]);虚子树变化
			c[x][1]=y;pushup(x);
		}
	}
	int findroot(int x)
	{
		access(x);splay(x);pushdown(x);
		while (c[x][0]) pushdown(x=c[x][0]);
		splay(x);
		return x;
	}
	void split(int x,int y)//x 为树新根，y 为 splay 新根
	{
		makeroot(x);
		access(y);
		splay(y);
	}
	void makeroot(int x)
	{
		access(x);splay(x);
		swp(x);
	}
	void link(int x,int y)//y 为新根
	{
		makeroot(x);
		if (x!=findroot(y))//可能已经连通
		{
			makeroot(y);f[x]=y;//虚子树变化
		}
	}
	void cut(int x,int y)
	{
		makeroot(x);
		if (x==findroot(y))//可能本不连通
		{
			pushdown(x);
			if (c[x][1]==y&&!c[y][0]&&!c[y][1])//可能连通但无边
			{
				c[x][1]=f[y]=0;//可能需要修改
				pushup(x);
			}
		}
	}
};
```

#### 带子树的 LCT

$O(n\log n)$，$O(n)$。

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
template<int N> struct LCT
{
	ll s[N],v[N],sg[N];
	int f[N],c[N][2],siz[N],st[N];
	//heap g[N]; //虚子树
	bool lz[N];
	void init(int n)
	{
		memset(f,0,n+1<<2);
		memset(c,0,n+1<<3);
		memset(s,0,n+1<<3);
		memset(v,0,n+1<<3);
		memset(lz,0,n+1);
	}
	bool nroot(int x)
	{
		return c[f[x]][0]==x||c[f[x]][1]==x;
	}
	void pushup(int x)
	{
		s[x]=s[c[x][0]]+s[c[x][1]]+v[x]+sg[x];
		siz[x]=siz[c[x][0]]+siz[c[x][1]]+1;
	}
	void pushdown(int x)
	{
		if (lz[x])
		{
			swap(c[c[x][0]][0],c[c[x][0]][1]);
			swap(c[c[x][1]][0],c[c[x][1]][1]);
			lz[c[x][0]]^=1;
			lz[c[x][1]]^=1;
			lz[x]=0;
		}
	}
	void zigzag(int x)
	{
		int y=f[x],z=f[y],typ=(c[y][0]==x);
		if (nroot(y)) c[z][c[z][1]==y]=x;
		f[x]=z;f[y]=x;
		if (c[x][typ]) f[c[x][typ]]=y;
		c[y][typ^1]=c[x][typ];c[x][typ]=y;
		pushup(y);
	}
	void splay(int x)
	{
		int y,tp=0;
		st[tp=1]=y=x;
		while (nroot(y)) st[++tp]=y=f[y];
		while (tp) pushdown(st[tp--]);
		for (;nroot(x);zigzag(x)) if (!nroot(f[x])) continue; else zigzag((c[f[x]][0]==x)^(c[f[f[x]]][0]==f[x]) ? x:f[x]);
		pushup(x);
	}
	void access(int x)
	{
		for (int y=0;x;x=f[y=x])
		{
			splay(x);sg[x]-=s[y];s[x]-=s[y];
			sg[x]+=s[c[x][1]];s[x]+=s[c[x][1]];
			//g[x].ins(s[c[x][1]]);g[x].del(s[y]);虚子树变化
			c[x][1]=y;pushup(x);
		}
	}
	int findroot(int x)
	{
		access(x);splay(x);pushdown(x);
		while (c[x][0]) pushdown(x=c[x][0]);
		splay(x);
		return x;
	}
	void split(int x,int y)
	{
		makeroot(x);
		access(y);
		splay(y);
	}
	void makeroot(int x)
	{
		access(x);splay(x);lz[x]^=1;swap(c[x][0],c[x][1]); pushup(x);
	}
	void link(int x,int y)
	{
		makeroot(x);
		if (x!=findroot(y))//可能已经连通
		{
			makeroot(y);f[x]=y;//虚子树变化
			sg[y]+=s[x];s[y]+=s[x];
		}
	}
	void cut(int x,int y)
	{
		makeroot(x);
		if (x==findroot(y))//可能本不连通
		{
			pushdown(x);
			if (c[x][1]==y&&!c[y][0]&&!c[y][1])//可能连通但无边
			{
				c[x][1]=f[y]=0;//可能需要修改
				pushup(x);
			}
		}
	}
};
const int N=2e5+2;
LCT<N> s;
int n,q,i,x,y,z,w;
void read(int &x)
{
	int c=getchar();
	while (c<48||c>57) c=getchar();
	x=c^48;c=getchar();
	while (c>=48&&c<=57) x=x*10+(c^48),c=getchar();
}
int main()
{
	read(n);read(q);s.init(n);
	for (i=1;i<=n;i++) read(x),s.s[i]=s.v[i]=x;
	for (i=1;i<n;i++)
	{
		read(x);read(y);++x;++y;
		s.link(x,y);
	}
	while (q--)
	{
		read(x);read(y);read(z);++y;
		if (x==0)
		{
			read(x);read(w);
			++z;++x;++w;
			s.cut(y,z);s.link(x,w);
			continue;
		}
		if (x==1)
		{
			s.split(y,y);
			s.s[y]=(s.v[y]+=z);
		}
		else
		{
			++z;
			s.split(y,z);
			printf("%lld\n",s.s[y]);
		}
	}
}
```

#### 换根树剖

$O(n+q\log n)$，$O(n)$。

```cpp
void dfs1(int x)
{
	int i;
	siz[x]=1;
	for (i=fir[x];i;i=nxt[i]) if (lj[i]!=f[x])
	{
		dep[lj[i]]=dep[f[lj[i]]=x]+1;
		dfs1(lj[i]);
		siz[x]+=siz[lj[i]];
		if (siz[hc[x]]<siz[lj[i]]) hc[x]=lj[i];
	}
}
void dfs2(int x)
{
	nfd[dfn[x]=++bs]=x;
	if (hc[x])
	{
		int i;
		top[hc[x]]=top[x];
		dfs2(hc[x]);
		for (i=fir[x];i;i=nxt[i]) if ((lj[i]!=f[x])&&(lj[i]!=hc[x])) dfs2(top[lj[i]]=lj[i]);
	}
}
void mdf(int xx,int yy)
{
	while (top[xx]!=top[yy])
	{
		if (dep[top[xx]]<dep[top[yy]]) swap(xx,yy);
		z=dfn[top[xx]];y=dfn[xx];xdsmdf(1);
		xx=f[top[xx]];
	}
	if (dep[xx]>dep[yy]) swap(xx,yy);
	z=dfn[xx];y=dfn[yy];
	xdsmdf(1);
}
int find(int x,int y)
{
	while ((top[x]!=top[y])&&(f[top[x]]!=y)) x=f[top[x]];
	if (top[x]==top[y]) return hc[y];
	return top[x];
}
int main()
{
	read(n);read(m);
	for (i=2;i<=n;i++)
	{
		read(x);read(y);
		add();
	}bs=0;
	for (i=1;i<=n;i++) read(v[i]);
	dfs1(dep[1]=1);dfs2(top[1]=1);
	read(rt);r[l[1]=1]=n;build(1);
	while (m--)
	{
		read(x);read(y);
		if (x==1) {rt=y;continue;}
		if (x==2)
		{
			read(x);read(dt);
			mdf(x,y);continue;
		}
		x=y;dt=inf;
		if (x==rt)
		{
			z=1;y=n;sum(1);
		}
		else if ((dfn[x]<dfn[rt])&&(dfn[x]+siz[x]>dfn[rt]))
		{
			c=find(rt,x);
			z=1;y=dfn[c]-1;if (z<=y) sum(1);
			z=dfn[c]+siz[c];y=n;if (z<=y) sum(1);
		}
		else
		{
			z=dfn[x];y=z+siz[x]-1;sum(1);
		}
		printf("%d\n",dt);
	}
}
```

#### 树上启发式合并，DSU on tree

```cpp
void dfs1(int x)
{
	siz[x]=zdep[x]=1;
	int i;
	for (i=fir[x];i;i=nxt[i]) if (lj[i]!=f[x])
	{
		dep[lj[i]]=dep[f[lj[i]]=x]+1;
		dfs1(lj[i]);
		siz[x]+=siz[lj[i]];
		if (siz[hc[x]]<siz[lj[i]]) hc[x]=lj[i];
		zdep[x]=max(zdep[x],zdep[lj[i]]+1);
	}
}
void cal(int x)
{
	int i;
	dl[tou=wei=1]=x;
	while (tou<=wei)
	{
		++dp[dep[x=dl[tou++]]];
		if ((dp[dep[x]]>dp[zd])||(dp[dep[x]]==dp[zd])&&(dep[x]<zd)) zd=dep[x];
		for (i=fir[x];i;i=nxt[i]) if (lj[i]!=f[x]) dl[++wei]=lj[i];
	}
}
void dfs2(int x)
{
	if (!hc[x])
	{
		if (++dp[dep[x]]>dp[zd]) zd=dep[x];
		return;
	}
	int i;
	for (i=fir[x];i;i=nxt[i]) if ((lj[i]!=f[x])&&(lj[i]!=hc[x]))
	{
		dfs2(lj[i]);
		memset(dp+dep[lj[i]],0,zdep[lj[i]]<<2);
	}
	dfs2(hc[x]);
	dp[dep[x]]=1;
	if (dp[zd]<=1) zd=dep[x];
	for (i=fir[x];i;i=nxt[i]) if ((lj[i]!=f[x])&&(lj[i]!=hc[x])) cal(lj[i]);
	ans[x]=zd-dep[x];
}
```

#### 长链剖分（$k$ 级祖先）

$O(n+q)$，$O(n)$。

```cpp
void dfs1(int x)
{
	int i;
	for (i=1;i<=er[dep[x]-1];i++) f[x][i]=f[f[x][i-1]][i-1];md[x]=dep[x];
	for (i=fir[x];i;i=nxt[i]) {dep[lj[i]]=dep[x]+1;dfs1(lj[i]);if (md[lj[i]]>md[dc[x]]) dc[x]=lj[i];}
	if (dc[x]) md[x]=md[dc[x]];
}
void dfs2(int x)
{
	int i;
	if (dc[x])
	{
		top[dc[x]]=top[x];
		dfs2(dc[x]);
		for (i=fir[x];i;i=nxt[i]) if (lj[i]!=dc[x]) dfs2(top[lj[i]]=lj[i]);
	}
	if (x==top[x])
	{
		c=md[x]-dep[x];y=x;up[x].push_back(x);down[x].push_back(x);
		for (i=1;(i<=c)&&(y=f[y][0]);i++) up[x].push_back(y);y=x;
		for (i=1;i<=c;i++) down[x].push_back(y=dc[y]);
	}
}
int main()
{
	int n,q,ans=0,x,y,c,i;
	ll ta=0;
	read(n);read(q);read(s);
	for (i=1;i<=n;i++) {read(f[i][0]);if (f[i][0]==0) rt=i; else add(f[i][0],i);}
	for (i=2;i<=n;i++) er[i]=er[i>>1]+1;dep[rt]=1;
	dfs1(rt);dfs2(top[rt]=rt);
	for (i=1;i<=q;i++)
	{
		x=(get(s)^ans)%n+1;y=(get(s)^ans)%dep[x];
		if (y==0) {ans=x;ta^=(ll)i*ans;continue;}
		c=dep[x]-y;x=top[f[x][er[y]]];
		if (dep[x]>c) ans=up[x][dep[x]-c]; else ans=down[x][c-dep[x]];
		ta^=(ll)i*ans;
	}
	printf("%lld",ta);
}
```

#### 长链剖分（dp 合并）

$O(n)$，$O(n)$。

```cpp
void dfs1(int x)
{
	top[x]=1;
	for (int i=fir[x];i;i=nxt[i]) if (!top[lj[i]])
	{
		dfs1(lj[i]);
		if (len[lj[i]]>len[hc[x]]) hc[x]=lj[i];
	}
	len[x]=len[hc[x]]+1;top[hc[x]]=0;
}
void dfs2(int x)
{
	*f[x]=1;gs[x]=1;
	if (!hc[x]) return;
	ed[x]=1;f[hc[x]]=f[x]+1;
	for (int i=fir[x];i;i=nxt[i]) if (!ed[lj[i]]) dfs2(lj[i]);
	ans[x]=ans[hc[x]]+1;gs[x]=gs[hc[x]];
	if (gs[x]==1) ans[x]=0;
	for (int i=fir[x];i;i=nxt[i]) if ((!ed[lj[i]])&&(lj[i]!=hc[x]))
	{
		int v=lj[i],*p;
		for (int j=0;j<len[v];j++)
		{
			*(p=f[x]+j+1)+=*(f[v]+j);
			if (j+1==ans[x]) {gs[x]=*p;continue;}
			if ((*p>gs[x])||(*p==gs[x])&&(j+1<ans[x])) {gs[x]=*p;ans[x]=j+1;}
		}
	}
	gs[x]=*(f[x]+ans[x]);
	ed[x]=0;
}
```

#### 动态 dp（全局平衡二叉树）

$O((n+q)\log n)$，$O(n)$。

```cpp
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <fstream>
using namespace std;
const int N=1e6+2,M=6e7+2,INF=-1e9;
struct matrix
{
	int a[2][2];
};
matrix s[N],js;
matrix operator *(matrix x,matrix y)
{
	js.a[0][0]=max(x.a[0][0]+y.a[0][0],x.a[0][1]+y.a[1][0]);
	js.a[0][1]=max(x.a[0][0]+y.a[0][1],x.a[0][1]+y.a[1][1]);
	js.a[1][0]=max(x.a[1][0]+y.a[0][0],x.a[1][1]+y.a[1][0]);
	js.a[1][1]=max(x.a[1][0]+y.a[0][1],x.a[1][1]+y.a[1][1]);
	return js;
}
int st[N],c[N][2],hc[N],lj[N<<1],nxt[N<<1],fir[N],siz[N],v[N],g[N][2],fa[N],f[N],val[N];
int n,m,i,j,x,y,z,dtp,stp,tp,fh,bs,rt,aaa,la;
char dr[M+5],sc[M];
void pushup(int x)
{
	s[x].a[0][0]=s[x].a[0][1]=g[x][0];
	s[x].a[1][0]=g[x][1];s[x].a[1][1]=INF;
	if (c[x][0]) s[x]=s[c[x][0]]*s[x];
	if (c[x][1]) s[x]=s[x]*s[c[x][1]];
}
void read(int &x)
{
	++dtp;fh=0;
	while ((dr[dtp]<48)||(dr[dtp]>57))
	{
		if (dr[dtp++]=='-')
		{
			fh=1;
			break;
		}
	}
	x=dr[dtp++]^48;
	while ((dr[dtp]>=48)&&(dr[dtp]<=57)) x=x*10+(dr[dtp++]^48);
	if (fh) x=-x;
}
void add(int x,int y)
{
	lj[++bs]=y;
	nxt[bs]=fir[x];
	fir[x]=bs;
	lj[++bs]=x;
	nxt[bs]=fir[y];
	fir[y]=bs;
}
bool nroot(int x)
{
	return ((c[f[x]][0]==x)||(c[f[x]][1]==x));
}
void dfs1(int x)
{
	siz[x]=1;
	int i;
	for (i=fir[x];i;i=nxt[i]) if (lj[i]!=fa[x])
	{
		fa[lj[i]]=x;
		dfs1(lj[i]);
		siz[x]+=siz[lj[i]];
		if (siz[hc[x]]<siz[lj[i]]) hc[x]=lj[i];
	}
}
int build(int l,int r)
{
	if (l>r) return 0;
	int i,tot=0,upn=0;
	for (i=l;i<=r;i++) tot+=val[i];tot>>=1;
	for (i=l;i<=r;i++)
	{
		upn+=val[i];
		if (upn>=tot)
		{
			f[c[st[i]][0]=build(l,i-1)]=st[i];
			f[c[st[i]][1]=build(i+1,r)]=st[i];
			pushup(st[i]);
			++aaa;
			return st[i];
		}
	}
}
int dfs2(int x)
{
	int i,j;
	for (i=x;i;i=hc[i]) for (j=fir[i];j;j=nxt[j]) if ((lj[j]!=fa[i])&&(lj[j]!=hc[i]))
	{
		f[y=dfs2(lj[j])]=i;
		g[i][0]+=max(s[y].a[0][0],s[y].a[1][0]);
		g[i][1]+=s[y].a[0][0];
	}
	tp=0;
	for (i=x;i;i=hc[i]) st[++tp]=i;
	for (i=1;i<tp;i++) val[i]=siz[st[i]]-siz[st[i+1]];
	val[tp]=siz[st[tp]];
	return build(1,tp);
}
void change(int x,int y)
{
	g[x][1]+=y-v[x];v[x]=y;
	while (f[x])
	{
		if (nroot(x)) pushup(x);
		else
		{
			g[f[x]][0]-=max(s[x].a[0][0],s[x].a[1][0]);
			g[f[x]][1]-=s[x].a[0][0];
			pushup(x);
			g[f[x]][0]+=max(s[x].a[0][0],s[x].a[1][0]);
			g[f[x]][1]+=s[x].a[0][0];
		}
		x=f[x];
	}
	pushup(x);
}
int main()
{
	scanf("%d%d",&n,&m);
	fread(dr+1,1,min(M,n*20+m*20),stdin);
	for (i=1;i<=n;i++)
	{
		read(g[i][1]);
		v[i]=g[i][1];
	}
	for (i=1;i<n;i++)
	{
		read(x);read(y);
		add(x,y);
	}
	dfs1(1);
	rt=dfs2(1);tp=0;
	while (m--)
	{
		read(x);read(y);
		change(x^la,y);
		x=la=max(s[rt].a[0][0],s[rt].a[1][0]);
		while (x)
		{
			st[++tp]=x%10;
			x/=10;
		}
		while (tp) sc[++stp]=st[tp--]|48;
		sc[++stp]=10;
	}
	fwrite(sc+1,1,stp,stdout);
}
```

#### 全局平衡二叉树（修改版）

$O((n+q)\log n)$，$O(n)$。

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int,int> pa;
const int N=1e6+2,M=1e6+2;
ll ans;
pa w[N];
int c[N][2],f[N],fa[N],v[N],s[N],lz[N],lj[M],nxt[M],siz[N],hc[N],fir[N],st[N];
int a[N],top[N];
int n,i,x,y,z,bs,tp,rt;
void read(int &x)
{
	int c=getchar();
	while (c<48||c>57) c=getchar();
	x=c^48;c=getchar();
	while (c>=48&&c<=57) x=x*10+(c^48),c=getchar();
}
void add()
{
	lj[++bs]=y;nxt[bs]=fir[x];fir[x]=bs;
	lj[++bs]=x;nxt[bs]=fir[y];fir[y]=bs;
}
void pushup(int &x)
{
	s[x]=min(v[x],min(s[c[x][0]],s[c[x][1]]));
}
void pushdown(int &x)
{
	if (lz[x]<0)
	{
		int cc=c[x][0];
		if (cc)
		{
			lz[cc]+=lz[x];s[cc]+=lz[x];v[cc]+=lz[x];
		}
		cc=c[x][1];
		if (cc)
		{
			v[cc]+=lz[x];lz[cc]+=lz[x];s[cc]+=lz[x];
		}lz[x]=0;
		return;
	}
}
bool nroot(int &x) {return c[f[x]][0]==x||c[f[x]][1]==x;}
bool cmp(pa &o,pa &p) {return o>p;}
void dfs1(int x)
{
	siz[x]=1;
	for (int i=fir[x];i;i=nxt[i]) if (lj[i]!=fa[x])
	{
		fa[lj[i]]=x;dfs1(lj[i]);siz[x]+=siz[lj[i]];
		if (siz[hc[x]]<siz[lj[i]]) hc[x]=lj[i];
	}
}
int build(int l,int r)
{
	if (l>r) return 0;
	if (l==r)
	{
		l=st[l];s[l]=v[l]=siz[l]>>1;
		return l;
	}
	int x=lower_bound(a+l,a+r+1,a[l]+a[r]>>1)-a,y=st[x];
	c[y][0]=build(l,x-1);
	c[y][1]=build(x+1,r);
	v[y]=siz[y]>>1;
	if (c[y][0]) f[c[y][0]]=y;
	if (c[y][1]) f[c[y][1]]=y;
	pushup(y);
	return y;
}
void dfs2(int x)
{
	if (!hc[x]) return;
	int i;
	top[hc[x]]=top[x];
	if (top[x]==x)
	{
		st[tp=1]=x;
		for (i=hc[x];i;i=hc[i]) st[++tp]=i;
		for (i=1;i<=tp;i++) a[i]=siz[st[i]]-siz[hc[st[i]]]+a[i-1];
		f[build(1,tp)]=fa[x];
	}
	dfs2(hc[x]);
	for (i=fir[x];i;i=nxt[i]) if (lj[i]!=fa[x]&&lj[i]!=hc[x]) dfs2(top[lj[i]]=lj[i]);
}
void mdf(int x)
{
	int y=x;
	st[tp=1]=x;
	while (y=f[y]) st[++tp]=y;y=x;
	while (tp) pushdown(st[tp--]);
	while (x)
	{
		--v[x];--lz[c[x][0]];--v[c[x][0]];--s[c[x][0]];
		while (c[f[x]][0]==x) x=f[x];x=f[x];
	}
	pushup(y);
	while (y=f[y]) pushup(y);
}
int ask(int x)
{
	int y=x;
	st[tp=1]=x;
	while (y=f[y]) st[++tp]=y;
	while (tp) pushdown(st[tp--]);
	int r=v[x];
	while (x)
	{
		r=min(r,min(v[x],s[c[x][0]]));
		while (c[f[x]][0]==x) x=f[x];x=f[x];
	}
	return r;
}
signed main()
{
	read(n);s[0]=1e9;
	for (i=1;i<=n;i++) read(w[w[i].second=i].first);
	for (i=1;i<n;i++) read(x),read(y),add();
	sort(w+1,w+n+1,cmp);dfs1(1);dfs2(top[1]=1);rt=1;while (f[rt]) rt=f[rt];
	for (i=1;i<=n&&v[rt];i++) if (ask(w[i].second)) mdf(w[i].second),ans+=w[i].first;
	printf("%lld",ans);
}
```

#### 单调队列优化树上背包

```cpp
#include <stdio.h>
#include <string.h>
#include <algorithm>
using namespace std;
const int N=502,M=4002,inf=-1e9;
int lj[N<<1],nxt[N<<1],fir[N],siz[N],v[N],p[N],l[N],f[N][M],num[M],dl[M];
int n,m,i,j,x,y,c,bs,t,ksiz,rt,zx,ans,tou,wei;
bool ed[N];
void add()
{
	lj[++bs]=y;
	nxt[bs]=fir[x];
	fir[x]=bs;
	lj[++bs]=x;
	nxt[bs]=fir[y];
	fir[y]=bs;
}
void read(int &x)
{
	c=getchar();
	while ((c<48)||(c>57)) c=getchar();
	x=c^48;c=getchar();
	while ((c>=48)&&(c<=57))
	{
		x=x*10+(c^48);
		c=getchar();
	}
}
void dfs2(int x)
{
	ed[x]=siz[x]=1;
	int i,zd=0;
	for (i=fir[x];i;i=nxt[i]) if (!ed[lj[i]])
	{
		dfs2(lj[i]);
		siz[x]+=siz[lj[i]];
		zd=max(zd,siz[lj[i]]);
	}
	zd=max(zd,ksiz-siz[x]);
	if (zd<zx)
	{
		zx=zd;
		rt=x;
	}
	ed[x]=0;
	for (i=1;i<=m;i++) f[x][i]=inf;
}
void dfs3(int x)
{
	if (p[x]>m) return;
	ed[x]=1;
	f[x][0]=max(f[x][0],0);
	int i;
	if (!l[x])
	{
		for (i=fir[x];i;i=nxt[i]) if (!ed[lj[i]])
		{
			for (j=0;j<=m;j++) f[lj[i]][j]=f[x][j];
			dfs3(lj[i]);
			for (j=m-p[x];~j;j--) f[x][j]=max(f[x][j],f[lj[i]][j]);
		}
		for (i=m;i>=p[x];i--) f[x][i]=f[x][i-p[x]]+v[x];
		for (i=0;i<p[x];i++) f[x][i]=inf;
		ed[x]=0;
		return;
	}
	for (i=0;i<p[x];i++)
	{
		y=(m-i)/p[x];
		num[dl[tou=wei=1]=0]=f[x][i];
		for (j=1;j<=y;j++)
		{
			while ((tou<wei)&&(j-dl[tou]>l[x])) ++tou;
			f[x][i+j*p[x]]=max(num[j]=f[x][i+j*p[x]],num[dl[tou]]+(j-dl[tou])*v[x]);
			while ((tou<=wei)&&(num[dl[wei]]+(j-dl[wei])*v[x]<=num[j])) --wei;
			dl[++wei]=j;
		}
	}
	for (i=fir[x];i;i=nxt[i]) if (!ed[lj[i]])
	{
		for (j=0;j<=m;j++) f[lj[i]][j]=f[x][j];
		dfs3(lj[i]);
		for (j=m-p[x];~j;j--) f[x][j]=max(f[x][j],f[lj[i]][j]);
	}
	for (i=m;i>=p[x];i--) f[x][i]=f[x][i-p[x]]+v[x];
	for (i=0;i<p[x];i++) f[x][i]=inf;
	ed[x]=0;
}
void dfs1(int x)
{
	int i,j=ksiz;
	rt=x;zx=n;
	dfs2(x);
	dfs3(x=rt);
	for (i=p[x];i<=m;i++) ans=max(ans,f[x][i]);
	ed[x]=1;
	for (i=fir[x];i;i=nxt[i]) if (!ed[lj[i]])
	{
		if (j>siz[lj[i]]) ksiz=siz[lj[i]]; else ksiz=j-siz[x];
		dfs1(lj[i]);
	}
}
int main()
{
	read(t);
	while (t--)
	{
		ans=0;
		read(n);read(m);
		for (i=1;i<=n;i++) read(v[i]);
		for (i=1;i<=n;i++) read(p[i]);
		for (i=1;i<=n;i++) read(l[i]);
		for (i=1;i<=n;i++) --l[i];
		memset(f,0,sizeof(f));
		ksiz=n;
		for (i=1;i<n;i++)
		{
			read(x);read(y);add();
		}
		dfs1(1);
		printf("%d\n",ans);
		memset(fir,0,sizeof(fir));bs=0;
		memset(ed,0,sizeof(ed));
	}
}
```

#### 树上背包

```cpp
void dfs(int x)
{
	int i;
	for (i=fir[x];i;i=nxt[i])
	{
		for (j=1;j<=m;j++) f[lj[i]][j]=f[x][j-1]+v[lj[i]];
		dfs(lj[i]);
		for (j=0;j<=m;j++) f[x][j]=max(f[x][j],f[lj[i]][j]);
	}
}
```

#### 虚树

$O(n+\sum k\log n)$，$O(n)$。

```cpp
void ins(int x)
{
	if (tp==0)
	{
		st[tp=1]=x;
		return;
	}
	ance=lca(st[tp],x);
	while (tp>1&&dep[ance]<dep[st[tp-1]])
	{
		add(st[tp-1],st[tp]);
		--tp;
	}
	if (dep[ance]<dep[st[tp]]) add(ance,st[tp--]);
	if (!tp||st[tp]!=ance) st[++tp]=ance;
	st[++tp]=x;
}
	sort(a+1,a+m+1,cmp);
	if (a[1]!=1) st[tp=1]=1;//先行添加根节点
    for (i=1;i<=m;i++) ins(a[i]);
    if (tp) while (--tp) add(st[tp],st[tp+1]);//回溯
```

#### 圆方树

$O(n+m)$，$O(n+m)$。

```cpp
#include <bits/stdc++.h>
using namespace std;
#if !defined(ONLINE_JUDGE)&&defined(LOCAL)
#include "my_header\debug.h"
#else
#define dbg(...); 1;
#endif
typedef unsigned int ui;
typedef long long ll;
#define all(x) (x).begin(),(x).end()
const int N=3e4+2,M=3e4+2;//M 包括方点
struct P
{
	int v,w,id;
	P(int a,int b,int c):v(a),w(b),id(c){}
};
struct Q
{
	int v,w;
	Q(int a,int b):v(a),w(b){}
};
vector<P> e[N];
vector<Q> fe[M];
int dfn[M],low[N],st[N],len[M],top[M],siz[M],hc[M],dep[M],f[M],rb[N];
bool ed[M];//ed,dfn,loop,sum,fe,hc,tp,id,cnt,dep[1] 需初始化（注意倍率），ed 大小为边数
int tp,id,cnt,n;
void dfs1(int u)
{
	dfn[u]=low[u]=++id;
	st[++tp]=u;
	for (auto [v,w,id]:e[u]) if (!ed[id])
	{
		if (dfn[v]) low[u]=min(low[u],dfn[v]),rb[v]=w; else
		{
			ed[id]=1;
			dfs1(v);
			if (dfn[u]>low[v]) low[u]=min(low[u],low[v]),rb[v]=w; else
			{
				int ntp=tp;
				while (st[ntp]!=v) --ntp;
				if (ntp==tp)//圆圆边
				{
					--tp;
					fe[u].emplace_back(v,w);
					f[v]=u;
					continue;
				}
				++cnt;f[cnt]=u;
				for (int i=ntp;i<=tp;i++) f[st[i]]=cnt;
				len[st[ntp]]=w;
				for (int i=ntp+1;i<=tp;i++) len[st[i]]=len[st[i-1]]+rb[st[i]];
				len[cnt]=len[st[tp]]+rb[u];
				fe[u].emplace_back(cnt,0);
				for (int i=ntp;i<=tp;i++) fe[cnt].emplace_back(st[i],min(len[st[i]],len[cnt]-len[st[i]]));
				tp=ntp-1;
			}
		}
	}
}
void dfs2(int u)
{
	siz[u]=1;
	for (auto [v,w]:fe[u])
	{
		dep[v]=dep[u]+w;
		dfs2(v);
		siz[u]+=siz[v];
		if (siz[v]>siz[hc[u]]) hc[u]=v;
	}
}
void dfs3(int u)
{
	dfn[u]=++id;
	if (hc[u])
	{
		top[hc[u]]=top[u];
		dfs3(hc[u]);
		for (auto [v,w]:fe[u]) if (v!=hc[u]) dfs3(top[v]=v);
	}
}
int lca(int u,int v)
{
	while (top[u]!=top[v]) if (dfn[top[u]]>dfn[top[v]]) u=f[top[u]]; else v=f[top[v]];//注意不能用 dep
	return dfn[u]<dfn[v]?u:v;
}
int find(int u,int v)//u 是根
{
	if (dfn[hc[u]]+siz[hc[u]]>dfn[v]) return hc[u];
	while (f[top[v]]!=u) v=f[top[v]];
	return top[v];
}
int dis(int u,int v)
{
	int o=lca(u,v),r=dep[u]+dep[v];
	if (o<=n) return r-(dep[o]<<1);
	u=find(o,u);v=find(o,v);
	if (len[u]>len[v]) swap(u,v);
	return r+min(len[v]-len[u],len[o]-(len[v]-len[u]))-dep[u]-dep[v];
}
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	int m,q,i;
	cin>>n>>m>>q;cnt=n;
	for (i=1;i<=m;i++)
	{
		int u,v,w;
		cin>>u>>v>>w;
		e[u].emplace_back(v,w,i);
		e[v].emplace_back(u,w,i);
	}
	mt19937 rnd(time(0));
	for (i=1;i<=n;i++) shuffle(all(e[i]),rnd);
	dfs1(1);id=0;
	dfs2(1);
	dfs3(top[1]=1);
	while (q--)
	{
		int u,v;
		cin>>u>>v;
		cout<<dis(u,v)<<'\n';
	}
}
```

#### 广义圆方树

$O(n+m)$，$O(n+m)$。

```cpp
void dfs(int x)
{
	int i;
	dfn[x]=low[x]=++sx;
	val[st[++top]=x]=-1;
	for (i=fir[x];i;i=next[i]) if (dfn[lj[i]]) low[x]=min(low[x],dfn[lj[i]]); else
	{
		dfs(lj[i]);
		low[x]=min(low[x],low[lj[i]]);
		if (dfn[x]<=low[lj[i]])
		{
			val[++fs]=1;
			while (st[top+1]!=lj[i])
			{
				++val[fs];
				fadd(fs,st[top--]);
			}
			fadd(fs,x);
		}
	}
}
```

#### 支配树（DAG 版）

$O(m\log n)$，$O(n\log n)$。

```cpp
int lca(int x,int y)
{
	int i;
	if (dep[x]<dep[y]) swap(x,y);
	for (i=lm[x];dep[x]!=dep[y];i--) if (dep[f[x][i]]>=dep[y]) x=f[x][i];
	if (x==y) return x;
	for (i=lm[x];f[x][0]!=f[y][0];i--) if (f[x][i]!=f[y][i])
	{
		x=f[x][i];y=f[y][i];
	}
	return f[x][0];
}
void dfs(int x)
{
	s[x]=1;
	int i;
	for (i=sfir[x];i;i=snxt[i])
	{
		dfs(slj[i]);
		s[x]+=s[slj[i]];
	}
}
int main()
{
	dep[0]=-1;
	read(n);
	for (i=1;i<=n;i++)
	{
		read(x);
		while (x)
		{
			add(x,i);
			read(x);
		}
	}
	dl[tou=wei=1]=++n;
	for (i=1;i<n;i++) if (!rd[i]) add(n,i);
	while (tou<=wei)
	{
		for (i=fir[x=dl[tou++]];i;i=nxt[i]) if (--rd[lj[i]]==0) dl[++wei]=lj[i];
		if (i=ffir[x])
		{
			y=flj[i];
			while (i=fnxt[i]) y=lca(y,flj[i]);
			f[x][0]=y;
		} else y=0;
		sadd(y,x);
		f[x][0]=y;
		for (i=1;i<=16;i++) if (0==(f[x][i]=f[f[x][i-1]][i-1]))
		{
			lm[x]=i;
			break;
		}
		dep[x]=dep[y]+1;
	}
	dfs(n);
	for (i=1;i<n;i++) printf("%d\n",s[i]-1);
}
```

#### 支配树（一般图）

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=2e5+2;
vector<int> lj[N],llj[N],fl[N],tl[N],buc[N],c[N];
int f[N],mn[N],siz[N],sdom[N],idom[N],dfn[N],nfd[N],pv[N];
int n,m,cnt,i,j,x,y,na;
bool reach[N];
void dfs1(int x)
{
	nfd[dfn[x]=++cnt]=x;
	for (auto v:lj[x]) if (!dfn[v]) dfs1(v),c[x].push_back(v);
}
int getf(int x)
{
	if (f[x]==x) return x;
	int u=getf(f[x]);
	mn[x]=dfn[sdom[mn[x]]]<dfn[sdom[mn[f[x]]]]?mn[x]:mn[f[x]];
	return f[x]=u;
}
void dfs0(int u)
{
	reach[u]=1;
	for (auto &v:lj[u]) if (!reach[v]) dfs0(v);
}
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	int S;
	cin>>n>>m>>S;++S;
	while (m--) cin>>x>>y,++x,++y,lj[x].push_back(y);
	for (i=1;i<=n;i++) mn[i]=f[i]=i;
	dfs0(S);
	for (i=1;i<=n;i++) if (reach[i]) for (auto &v:lj[i]) if (reach[v]) llj[i].push_back(v),fl[v].push_back(i);
	for (i=1;i<=n;i++) lj[i]=llj[i];
	dfs1(S);dfn[0]=1e9;
	for (i=cnt;i;i--)
	{
		x=nfd[i];na=0;
		for (auto v:fl[x])
		{
			sdom[x]=dfn[sdom[x]]<dfn[v]?sdom[x]:v;
			if (dfn[v]>dfn[x])
			{
				getf(v);
				na=dfn[sdom[na]]<dfn[sdom[mn[v]]]?na:mn[v];
			}
		}
		sdom[x]=dfn[sdom[x]]<dfn[sdom[na]]?sdom[x]:sdom[na];
		buc[sdom[x]].push_back(x);
		for (auto v:buc[x]) getf(v),pv[v]=mn[v];
		for (auto v:c[x]) f[v]=x,mn[v]=dfn[sdom[mn[v]]]<dfn[sdom[mn[x]]]?mn[v]:mn[x];
	}
	for (i=1;i<=n;i++) idom[nfd[i]]=(sdom[pv[nfd[i]]]==sdom[nfd[i]])?sdom[nfd[i]]:idom[pv[nfd[i]]];
	for (i=1;i<=n;i++) cout<<(i==S?S:idom[i])-1<<" \n"[i==n];
}
```

#### 最小树形图（朱刘算法，无方案）

$O(nm)$，$O(n+m)$。

```cpp
int main()
{
	read(n);read(m);read(rt);
	for (i=1;i<=m;i++)
	{
		read(lj[i][1]);read(lj[i][2]);read(lj[i][0]);
	}
	while (1)
	{
		memset(infl,0x3f,sizeof(infl));
		memset(ed,0,sizeof(ed));
		memset(fa,0,sizeof(fa));
		for (i=1;i<=m;i++) if ((lj[i][1]!=lj[i][2])&&(lj[i][2]!=rt)&&(infl[lj[i][2]]>lj[i][0]))
		{
			infl[lj[i][2]]=lj[i][0];
			pre[lj[i][2]]=lj[i][1];
		}
		for (i=1;i<=n;i++) if (i!=rt)
		{
			if (infl[i]==infl[0])
			{
				puts("-1");return 0;
			}
			ans+=infl[i];
			for (j=i;(ed[j]!=i)&&(fa[j]==0)&&(j!=rt);j=pre[j]) ed[j]=i;
			if (ed[j]==i)
			{
				++cnt;
				while (fa[j]==0)
				{
					fa[j]=cnt;
					j=pre[j];
				}
			}
		}
		if (!cnt)
		{
			printf("%d",ans);return 0;
		}
		for (i=1;i<=n;i++) if (!fa[i]) fa[i]=++cnt;
		for (i=1;i<=m;i++)
		{
			lj[i][0]-=infl[lj[i][2]];
			lj[i][1]=fa[lj[i][1]];
			lj[i][2]=fa[lj[i][2]];
		}
		rt=fa[rt];
		n=cnt;cnt=0;
	}
}
```

#### 最小乘积生成树

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=202,M=10002;
template<typename typC> void read(typC &x)
{
	int c=getchar(),fh=1;
	while ((c<48)||(c>57))
	{
		if (c=='-') {c=getchar();fh=-1;break;}
		c=getchar();
	}
	x=c^48;c=getchar();
	while ((c>=48)&&(c<=57))
	{
		x=x*10+(c^48);
		c=getchar();
	}
	x*=fh;
}
struct P
{
	int x,y;
	P(int a=0,int b=0):x(a),y(b){}
	bool operator<(const P &o) const {return (ll)x*y<(ll)o.x*o.y||(ll)x*y==(ll)o.x*o.y&&x<o.x;}
};
struct Q
{
	int u,v,x,y,val;
	bool operator<(const Q &o) const {return val<o.val;}
};
P ans=P(1e9,1e9),l,r;
Q a[M];
int f[N];
int n,m,i;
int getf(int x)
{
	if (f[x]==x) return x;
	return f[x]=getf(f[x]);
}
P sol1()
{
	P r=P(0,0);
	for (i=1;i<=n;i++) f[i]=i;
	sort(a+1,a+m+1);
	for (i=1;i<=m;i++) if (getf(a[i].u)!=getf(a[i].v))
	{
		f[f[a[i].u]]=f[a[i].v];
		r.x+=a[i].x,r.y+=a[i].y;
	}
	return r;
}
void sol2(P l,P r)
{
	for (i=1;i<=m;i++) a[i].val=(r.x-l.x)*a[i].y+(l.y-r.y)*a[i].x;
	P np=sol1();
	ans=min(ans,np);
	if ((ll)(r.x-l.x)*(np.y-l.y)-(ll)(r.y-l.y)*(np.x-l.x)>=0) return;
	sol2(l,np);sol2(np,r);
}
int main()
{
	read(n);read(m);
	for (i=1;i<=m;i++) read(a[i].u),read(a[i].v),read(a[i].x),read(a[i].y),++a[i].u,++a[i].v;
	for (i=1;i<=m;i++) a[i].val=a[i].x;l=sol1();
	for (i=1;i<=m;i++) a[i].val=a[i].y;r=sol1();
	ans=min(ans,min(l,r));sol2(l,r);
	printf("%d %d",ans.x,ans.y);
}
```

#### 最小斯坦纳树

$O(3^kn+2^km\log m)$。

```cpp
const int N=102,M=1002,K=1024;
typedef long long ll;
typedef pair<ll,int> pa;
priority_queue<pa,vector<pa>,greater<pa> > heap;
pa cr;
ll f[K][N],inf;
int lj[M],len[M],nxt[M],fir[N];
int n,m,q,i,j,k,x,y,z,bs,c;
void add()
{
	lj[++bs]=y;
	len[bs]=z;
	nxt[bs]=fir[x];
	fir[x]=bs;
	lj[++bs]=x;
	len[bs]=z;
	nxt[bs]=fir[y];
	fir[y]=bs;
}
void read(int &x)
{
	c=getchar();
	while ((c<48)||(c>57)) c=getchar();
	x=c^48;c=getchar();
	while ((c>=48)&&(c<=57))
	{
		x=x*10+(c^48);
		c=getchar();
	}
}
void dijk(int s)
{
	int i;
	while (!heap.empty())
	{
		x=heap.top().second;heap.pop();
		for (i=fir[x];i;i=nxt[i]) if (f[s][lj[i]]>f[s][x]+len[i])
		{
			cr.first=f[s][cr.second=lj[i]]=f[s][x]+len[i];
			heap.push(cr);
		}
		while ((!heap.empty())&&(heap.top().first!=f[s][heap.top().second])) heap.pop();
	}
}
int main()
{
	memset(f,0x3f,sizeof(f));inf=f[0][0];
	read(n);read(m);read(q);
	while (m--)
	{
		read(x);read(y);read(z);
		add();
	}
	for (i=1;i<=q;i++)
	{
		read(x);
		f[1<<i-1][x]=0;
	}
	q=(1<<q)-1;
	for (i=1;i<=q;i++)
	{
		for (k=1;k<=n;k++)
		{
			for (j=i&(i-1);j;j=i&(j-1)) f[i][k]=min(f[i][k],f[j][k]+f[i^j][k]);
			if (f[i][k]<inf) heap.push(pa(f[i][k],k));
		}
		dijk(i);
	}
	for (i=1;i<=n;i++) inf=min(inf,f[q][i]);
	printf("%lld",inf);
}
```

#### $2$-sat

$O(n+m)$，$O(n+m)$。

```cpp
#define R int
const int N=2e6+2;
int lj[N],fir[N],bs,nxt[N],low[N],dfn[N],num[N],ds,fs,st[N],tp;
bool ed[N];
void dfs(int x)
{
	low[x]=dfn[x]=++ds;
	ed[st[++tp]=x]=1;
	int i;
	for (i=fir[x];i;i=nxt[i]) if (dfn[lj[i]])
	{
		if (ed[lj[i]]) low[x]=min(low[x],dfn[lj[i]]);
	}
	else
	{
		dfs(lj[i]);
		low[x]=min(low[x],low[lj[i]]);
	}
	if (dfn[x]==low[x])
	{
		++fs;
		while (st[tp+1]!=x)
		{
			ed[st[tp]]=false;
			num[st[tp--]]=fs;
		}
	}
}
void add(int x,int y)
{
	lj[++bs]=y;
	nxt[bs]=fir[x];
	fir[x]=bs;
}
void read(int &x)
{
	int c=getchar();
	while (c<48||c>57) c=getchar();
	x=c^48;c=getchar();
	while (c>=48&&c<=57) x=x*10+(c^48),c=getchar();
}
int main()
{
	R n,m,i,j,a,b;
	read(n);read(m);
	while (m--)
	{
		read(i);read(a);read(j);read(b);//i 为 a 或 j 为 b
		add(i+(a^1)*n,j+b*n);
		add(j+(b^1)*n,i+a*n);
	}
	n<<=1;
	for (i=1;i<=n;i++) if (!dfn[i]) dfs(i);
	n>>=1;
	for (i=1;i<=n;i++) if (num[i]==num[i+n])
	{
		puts("IMPOSSIBLE");
		return 0;
	}
	puts("POSSIBLE");
	for (i=1;i<=n;i++)
	{
		putchar(48|(num[i]>num[i+n]));
		putchar(32);
	}
}
```

#### Kosaraju 强连通分量（bitset 优化）

$O(\frac{n^2}w)$，$O(\frac {n^2}w)$。

```cpp
void dfs1(int x)
{
	int i;ed[x]=0;
	for (i=(lj[x]&ed)._Find_first();i<=n;i=(lj[x]&ed)._Find_next(i)) dfs1(i);
	sx[--tp]=x;
}
void dfs2(int x)
{
	int i;ed[x]=0;tv[f[x]=f[0]]+=v[x];
	for (i=(fj[x]&ed)._Find_first();i<=n;i=(fj[x]&ed)._Find_next(i)) dfs2(i);
}
int main()
{
	read(n);read(m);tp=n+1;
	for (i=1;i<=n;i++) {ed[i]=1;read(v[i]);}
	for (i=1;i<=m;i++)
	{
		read(x);read(y);lj[x][y]=1;fj[y][x]=1;lb[i][0]=x;lb[i][1]=y;
	}
	for (i=1;i<=n;i++) if (ed[i]) dfs1(i);
	ed.set();
	for (i=1;i<=n;i++) if (ed[sx[i]]) {++f[0];dfs2(sx[i]);}
	for (i=1;i<=m;i++) if (f[lb[i][0]]!=f[lb[i][1]])
	{
		flj[f[lb[i][0]]].push_back(f[lb[i][1]]);++rd[f[lb[i][1]]];
	}
	for (i=1;i<=f[0];i++) if (!rd[i]) dl[++wei]=i;
	while (tou<=wei)
	{
		x=dl[tou++];g[x]+=tv[x];
		for (i=0;i<flj[x].size();i++)
		{
			g[flj[x][i]]=max(g[flj[x][i]],g[x]);
			if (--rd[flj[x][i]]==0) dl[++wei]=flj[x][i];
		}
	}
	for (i=1;i<=f[0];i++) ans=max(ans,g[i]);printf("%d",ans); 
}

```

#### Tarjan 强连通分量

$O(n+m)$，$O(n+m)$。

```cpp
void dfs(int x)
{
	low[x]=dfn[x]=++ds;
	ed[st[++tp]=x]=1;
	int i;
	for (i=fir[x];i;i=next[i]) if (dfn[lj[i]])
	{
		if (ed[lj[i]]) low[x]=min(low[x],dfn[lj[i]]);
	}
	else
	{
		dfs(lj[i]);
		low[x]=min(low[x],low[lj[i]]);
	}
	if (dfn[x]==low[x])
	{
		++fs;st[tp+1]=0;
		while (st[tp+1]!=x)
		{
			ed[st[tp]]=false;
			num[st[tp--]]=fs;
		}
	}
}
```

#### 欧拉回路构造

$O(n+m)$，$O(n+m)$。

```cpp
void dfs(int x)
{
	int y;
	while (ed[fir[x]]) fir[x]=nxt[fir[x]];
	if (!fir[x]) return;
	while (fir[x])
	{
		ed[fir[x]]=ed[fir[x]^t]=1;
		y=fir[x];fir[x]=nxt[fir[x]];
		while (ed[fir[x]]) fir[x]=nxt[fir[x]];
		dfs(lj[y]);
		st[++m]=num[y];
	}
}
int getf(int x)
{
	if (f[x]==x) return x;
	return f[x]=getf(f[x]);
}
int main()
{
	read(t);read(n);read(m);t&=1;
	if (m==0)
	{
		puts("YES");return 0;
	}
	if (t) for (i=1;i<=n;i++) f[i]=i;
	for (i=1;i<=m;i++)
	{
		read(x);read(y);
		++rd[y];if (t) ++rd[x]; else ++cd[x];
		lj[++bs]=y;
		num[bs]=i;
		nxt[bs]=fir[x];
		fir[x]=bs;
		if (t)
		{
			lj[++bs]=x;
			num[bs]=-i;
			nxt[bs]=fir[y];
			fir[y]=bs;
			f[getf(x)]=getf(y);
		}
	}
	bs=m=0;
	if (t)
	{
		for (i=1;i<=n;i++) if (rd[i]&1)
		{
			puts("NO");return 0;
		}
		for (i=1;i<=n;i++) if (rd[i])
		{
			if (!lst)
			{
				lst=i;
				continue;
			}
			if (getf(lst)!=getf(i))
			{
				puts("NO");return 0;
			}
			lst=i;
		}
		for (i=1;i<=n;i++) if (rd[i]) break;
	}
	else
	{
		for (i=1;i<=n;i++) if (rd[i]!=cd[i])
		{
			puts("NO");return 0;
		}
		for (i=1;i<=n;i++) if (rd[i]) break;
		f[dl[tou=wei=1]=i]=1;
		while (tou<=wei) for (i=fir[x=dl[tou++]];i;i=nxt[i]) if (!f[lj[i]]) f[dl[++wei]=lj[i]]=1;
		for (i=1;i<=n;i++) if (((rd[i])&&(!f[i])))
		{
			puts("NO");return 0;
		}
		for (i=1;i<=n;i++) if (rd[i]) break;
	}
	puts("YES");
	dfs(i);
	while (m) printf("%d ",st[m--]);
}
```

#### 有向图欧拉回路计数（BEST 定理）

$O(n^3)$，$O(n^2)$。

以 $u$ 为起点的欧拉回路个数 $sum=T(u)\times \prod\limits_{v=1}^n(out(v)-1)!$，其中 $T(u)$ 是以 $u$ 为根的外向树个数，$out(v)$ 是 $v$ 的出度。若允许循环同构（如 $1\to 2\to 1\to 3\to 1$ 与 $1\to 3\to 1\to 2\to 1$），还需多乘 $out(u)$。

```cpp
//https://blog.csdn.net/Jaihk662/article/details/79338437
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=102,M=4e5+2,p=1e6+3;
int a[N][N],fac[M],cd[N],st[N],rd[N],f[N],b[N][N];
int n,i,j,x,c,ans,t,tp;
void read(int &x)
{
	c=getchar();
	while ((c<48)||(c>57)) c=getchar();
	x=c^48;c=getchar();
	while ((c>=48)&&(c<=57))
	{
		x=x*10+(c^48);
		c=getchar();
	}
}
int ksm(int x,int y)
{
	int r=1;
	while (y)
	{
		if (y&1) r=(ll)r*x%p;
		x=(ll)x*x%p;
		y>>=1;
	}
	return r;
}
int GS()
{
	int i,j,k,r=1,xs;
	int fh=0;
	for (i=1;i<=n;i++)
	{
		k=0;
		for (j=i;j<=n;j++) if (a[j][i])
		{
			k=j;break;
		}
		assert(k);
		if (k==0) return 0;
		if (j!=k) fh^=1;
		for (j=i;j<=n;j++) swap(a[i][j],a[k][j]);
		xs=ksm(a[i][i],p-2);
		for (j=i+1;j<=n;j++) a[i][j]=(ll)a[i][j]*xs%p;
		r=(ll)r*a[i][i]%p;
		for (j=i+1;j<=n;j++)
		{
			xs=p-a[j][i];
			for (k=i+1;k<=n;k++) a[j][k]=(a[j][k]+(ll)(p-a[j][i])*a[i][k])%p;
		}
	}
	if (fh) return p-r; return r;
}
int getf(int x){if (f[x]==x) return x;return f[x]=getf(f[x]);}
int main()
{
	fac[0]=1;
	for (i=1;i<M;i++) fac[i]=(ll)fac[i-1]*i%p;
	read(t);
	while (t--)
	{
		read(n);
		memset(cd,0,sizeof(cd));
		memset(rd,0,sizeof(rd));
		memset(b,0,sizeof(b));
		for (i=1;i<=n;i++) f[i]=i;
		for (i=1;i<=n;i++)
		{
			read(cd[i]);
			for (j=1;j<=cd[i];j++)
			{
				read(x);--b[x][i];++b[i][i];++rd[x];
				f[getf(x)]=getf(i);
			}
		}
		for (i=1;i<=n;i++) if (rd[i]!=cd[i]) {puts("0");break;}
		if (i<=n) continue;
		tp=0;
		for (i=2;i<=n;i++) if (rd[i])
		{
			if (getf(i)!=getf(1)) {puts("0");break;}
			st[++tp]=i;
		}
		if (i<=n) continue;
		ans=1;if (cd[1]>1) ans=(ll)ans*cd[1]%p;
		for (i=1;i<=n;i++) if (cd[i]>2) ans=(ll)ans*fac[cd[i]-1]%p;
		//if (!cd[1]) {puts("1");continue;}
		for (i=1;i<=tp;i++) for (j=1;j<=tp;j++) a[i][j]=b[st[i]][st[j]];
		n=tp;
		for (i=1;i<=n;i++) for (j=1;j<=n;j++) if (a[i][j]<0) a[i][j]+=p;
		ans=(ll)ans*GS()%p;
		//if (1^n&1) ans=p-ans;
		printf("%d\n",ans%p);
	}
}
```

<div style="page-break-after:always"></div>

### 计算几何

#### 自适应 simpson 法

```cpp
const db eps=1e-7;
db sl,sr,sm,a;
db f(db x)
{
	return pow(x,a/x-x);
}
db g(db l,db r)
{
	db mid=(l+r)*0.5;
	return (f(l)+f(r)+f(mid)*4)/6*(r-l);
}
db ab(db x)
{
	if (x>0) return x;
	return -x;
}
db sim(db l,db r)
{
	db mid=(l+r)*0.5;
	sl=g(l,mid);sr=g(mid,r);sm=g(l,r);
	if (ab(sl+sr-sm)<eps) return sl+sr;
	return sim(l,mid)+sim(mid,r);
}
```

#### 板子

```cpp
#include <bits/stdc++.h>
using namespace std;
namespace geometry//不要用 int！
{
	#define tmpl template<typename T>
	typedef long long ll;
	typedef long double db;
	const db eps=1e-6;
	#define all(x) (x).begin(),(x).end()
	inline int sgn(const ll &x)
	{
		if (x<0) return -1;
		return x>0;
	}
	inline int sgn(const db &x)
	{
		if (fabs(x)<eps) return 0;
		return x>0?1:-1;
	}
	tmpl struct point//* 为叉乘，& 为点乘，只允许使用 double 和 ll
	{
		T x,y;
		point(){}
		point(T a,T b):x(a),y(b){}
		operator point<ll>() const {return point<ll>(x,y);}
		operator point<db>() const {return point<db>(x,y);}
		point<T> operator+(const point<T> &o) const {return point(x+o.x,y+o.y);}
		point<T> operator-(const point<T> &o) const {return point(x-o.x,y-o.y);}
		point<T> operator*(const T &k) const {return point(x*k,y*k);}
		point<T> operator/(const T &k) const {return point(x/k,y/k);}
		T operator*(const point<T> &o) const {return x*o.y-y*o.x;}
		T operator&(const point<T> &o) const {return (ll)x*o.x+(ll)y*o.y;}
		void operator+=(const point<T> &o) {x+=o.x;y+=o.y;}
		void operator-=(const point<T> &o) {x+=o.x;y+=o.y;}
		void operator*=(const T &k) {x*=k;y*=k;}
		void operator/=(const T &k) {x/=k;y/=k;}
		bool operator==(const point<T> &o) const {return x==o.x&&y==o.y;}
		bool operator!=(const point<T> &o) const {return x!=o.x||y!=o.y;}
		db len() const {return sqrt(len2());}//模长
		T len2() const {return x*x+y*y;}
	};
	const point<db> npos=point<db>(514e194,9810e191),apos=point<db>(145e174,999e180);
	const int DS[4]={1,2,4,3};
	tmpl int quad(const point<T> &o)//坐标轴归右上象限，返回值 [1,4]
	{
		return DS[(sgn(o.y)<0)*2+(sgn(o.x)<0)];
	}
	tmpl bool angle_cmp(const point<T> &a,const point<T> &b)
	{
		int c=quad(a),d=quad(b);
		if (c!=d) return c<d;
		return a*b>0;
	}
	tmpl db dis(const point<T> &a,const point<T> &b) {return (a-b).len();}
	tmpl T dis2(const point<T> &a,const point<T> &b) {return (a-b).len2();}
	tmpl point<T> operator*(const T &k,const point<T> &o) {return point<T>(k*o.x,k*o.y);}
	tmpl bool operator<(const point<T> &a,const point<T> &b)
	{
		int s=sgn(a*b);
		return s>0||s==0&&sgn(a.len2()-b.len2())<0;
	}
	tmpl istream & operator>>(istream &cin,point<T> &o) {return cin>>o.x>>o.y;}
	tmpl ostream & operator<<(ostream &cout,const point<T> &o) 
	{
		if ((point<db>)o==apos) return cout<<"all position";
		if ((point<db>)o==npos) return cout<<"no position";
		return cout<<'('<<o.x<<','<<o.y<<')';
	}
	tmpl struct line
	{
		point<T> o,d;
		line(){}
		line(const point<T> &a,const point<T> &b,int twopoint);
		bool operator!=(const line<T> &m) {return !(*this==m);}
	};
	template<> line<ll>::line(const point<ll> &a,const point<ll> &b,int twopoint)
	{
		o=a;
		d=twopoint?b-a:b;
		ll tmp=gcd(d.x,d.y);
		assert(tmp);
		if (d.x<0||d.x==0&&d.y<0) tmp=-tmp;
		d.x/=tmp;d.y/=tmp;
	}
	template<> line<db>::line(const point<db> &a,const point<db> &b,int twopoint)
	{
		o=a;
		d=twopoint?b-a:b;
		int s=sgn(d.x);
		if (s<0||!s&&d.y<0) d.x=-d.x,d.y=-d.y;
	}
	tmpl line<T> rotate_90(const line<T> &m) {return line(m.o,point(m.d.y,-m.d.x),0);}
	tmpl line<db> rotate(const line<T> &m,db angle)
	{
		return {(point<db>)m.o,{m.d.x*cos(angle)-m.d.y*sin(angle),m.d.x*sin(angle)+m.d.y*cos(angle)},0};
	}
	tmpl db get_angle(const line<T> &m,const line<T> &n) {return asin((m.d*n.d)/(m.d.len()*n.d.len()));}
	tmpl bool operator<(const line<T> &m,const line<T> &n)
	{
		int s=sgn(m.d*n.d);
		return s?s>0:m.d*m.o<n.d*n.o;
	}
	bool operator==(const line<ll> &m,const line<ll> &n) {return m.d==n.d&&(m.o-n.o)*m.d==0;}
	bool operator==(const line<db> &m,const line<db> &n) {return fabs(m.d*n.d)<eps&&fabs((n.o-m.o)*m.d)<eps;}
	tmpl ostream & operator<<(ostream &cout,const line<T> &o) {return cout<<'('<<o.d.x<<" k + "<<o.o.x<<" , "<<o.d.y<<" k + "<<o.o.y<<")";}
	tmpl point<db> intersect(const line<T> &m,const line<T> &n)
	{
		if (!sgn(m.d*n.d))
		{
			if (!sgn(m.d*(n.o-m.o))) return apos;
			return npos;
		}
		return (point<db>)m.o+(n.o-m.o)*n.d/(db)(m.d*n.d)*(point<db>)m.d;
	}
	tmpl db dis(const line<T> &m,const point<T> &o) {return m.d*(o-m.o)/m.d.len();}
	tmpl db dis(const point<T> &o,const line<T> &m) {return m.d*(o-m.o)/m.d.len();}
	struct circle
	{
		point<db> o;
		db r;
		circle(){}
		circle(const point<db> &O,const db &R=0):o(point<db>((db)O.x,(db)O.y)),r(R){}//圆心半径构造
		circle(const point<db> &a,const point<db> &b)//直径构造
		{
			o=(a+b)*0.5;
			r=dis(b,o);
		}
		circle(const point<db> &a,const point<db> &b,const point<db> &c)//三点构造外接圆（非最小圆）
		{
			auto A=(b+c)*0.5,B=(a+c)*0.5;
			o=intersect(rotate_90(line(A,c,1)),rotate_90(line(B,c,1)));
			r=dis(o,c);
		}
		circle(vector<point<db>> a)
		{
			int n=a.size(),i,j,k;
			mt19937 rnd(75643);
			shuffle(all(a),rnd);
			*this=circle(a[0]);
			for (i=1;i<n;i++) if (!cover(a[i]))
			{
				*this=circle(a[i]);
				for (j=0;j<i;j++) if (!cover(a[j]))
				{
					*this=circle(a[i],a[j]);
					for (k=0;k<j;k++) if (!cover(a[k])) *this=circle(a[i],a[j],a[k]);
				}
			}
		}
		circle(const vector<point<ll>> &b)
		{
			vector<point<db>> a(b.size());
			int n=a.size(),i,j,k;
			for (i=0;i<a.size();i++) a[i]=(point<db>)b[i];
			*this=circle(a);
		}
		tmpl bool cover(const point<T> &a) {return sgn(dis((point<db>)a,o)-r)<=0;}
	};
	tmpl struct segment
	{
		point<T> a,b;
		segment(){}
		segment(point<T> o,point<T> p)
		{
			int s=sgn(o.x-p.x);
			if (s>0||!s&&o.y>p.y) swap(o,p);
			a=o;b=p;
		}
	};
	tmpl bool intersect(const segment<T> &m,const segment<T> &n)
	{
		auto a=n.b-n.a,b=m.b-m.a;
		auto d=n.a-m.a;
		if (sgn(n.b.x-m.a.x)<0||sgn(m.b.x-n.a.x)<0) return 0;
		if (sgn(max(n.a.y,n.b.y)-min(m.a.y,m.b.y))<0||sgn(max(m.a.y,m.b.y)-min(n.a.y,n.b.y))<0) return 0;
		return sgn(b*d)*sgn((n.b-m.a)*b)>=0&&sgn(a*d)*sgn((m.b-n.a)*a)<=0;
	}
	tmpl struct convex
	{
		vector<point<T>> p;
		convex(vector<point<T>> a);
		db peri()//周长
		{
			int i,n=p.size();
			db C=(p[n-1]-p[0]).len();
			for (i=1;i<n;i++) C+=(p[i-1]-p[i]).len();
			return C;
		}
		db area(){return area2()*0.5;}//面积
		T area2()//两倍面积
		{
			int i,n=p.size();
			T S=p[n-1]*p[0];
			for (i=1;i<n;i++) S+=p[i-1]*p[i];
			return abs(S);
		}
		db diam() {return sqrt(diam2());}
		T diam2()//直径平方
		{
			T r=0;
			int n=p.size(),i,j;
			if (n<=2)
			{
				for (i=0;i<n;i++) for (j=i+1;j<n;j++) r=max(r,dis2(p[i],p[j]));
				return r;
			}
			p.push_back(p[0]);
			for (i=0,j=1;i<n;i++)
			{
				while ((p[i+1]-p[i])*(p[j]-p[i])<=(p[i+1]-p[i])*(p[j+1]-p[i])) if (++j==n) j=0;
				r=max({r,dis2(p[i],p[j]),dis2(p[i+1],p[j])});
			}
			p.pop_back();
			return r;
		}
		bool cover(const point<T> &o) const//点是否在凸包内
		{
			if (o.x<p[0].x||o.x==p[0].x&&o.y<p[0].y) return 0;
			if (o==p[0]) return 1;
			if (p.size()==1) return 0;
			ll tmp=(o-p[0])*(p.back()-p[0]);
			if (tmp==0) return dis2(o,p[0])<=dis2(p.back(),p[0]);
			if (tmp<0||p.size()==2) return 0;
			int x=upper_bound(1+all(p),o,[&](const point<T> &a,const point<T> &b){return (a-p[0])*(b-p[0])>0;})-p.begin()-1;
			return (o-p[x])*(p[x+1]-p[x])<=0;
		}
		convex<T> operator+(const convex<T> &A) const
		{
			int n=p.size(),m=A.p.size(),i,j;
			vector<point<T>> c;
			if (min(n,m)<=2)
			{
				c.reserve(n*m);
				for (i=0;i<n;i++) for (j=0;j<m;j++) c.push_back(p[i]+A.p[j]);
				return convex<T>(c);
			}
			point<T> a[n],b[m];
			for (i=0;i+1<n;i++) a[i]=p[i+1]-p[i];
			a[n-1]=p[0]-p[n-1];
			for (i=0;i+1<m;i++) b[i]=A.p[i+1]-A.p[i];
			b[m-1]=A.p[0]-A.p[m-1];
			c.reserve(n+m);
			c.push_back(p[0]+A.p[0]);
			for (i=j=0;i<n&&j<m;) c.push_back(c.back()+(a[i]*b[j]>0?a[i++]:b[j++]));
			while (i<n-1) c.push_back(c.back()+a[i++]);
			while (j<m-1) c.push_back(c.back()+b[j++]);
			return convex<T>(c);
		}
		void operator+=(const convex &a) {*this=*this+a;}
	};
	tmpl convex<T>::convex(vector<point<T>> a)
	{
		int n=a.size(),i;
		if (!n) return;
		p=a;
		for (i=1;i<n;i++) if (p[i].x<p[0].x||p[i].x==p[0].x&&p[i].y<p[0].y) swap(p[0],p[i]);
		a.resize(0);a.reserve(n);
		for (i=1;i<n;i++) if (p[i]!=p[0]) a.push_back(p[i]-p[0]);
		sort(all(a));
		for (i=0;i<n;i++) a[i]+=p[0];
		point<T>* st=p.data()-1;
		int tp=1;
		for (auto &v:a)
		{
			while (tp>1&&sgn((st[tp]-st[tp-1])*(v-st[tp-1]))<=0) --tp;
			st[++tp]=v;
		}
		p.resize(tp);
	}
	template<> bool convex<db>::cover(const point<db> &o) const//点是否在凸包内
	{
		if (o.x<p[0].x||o.x==p[0].x&&o.y<p[0].y) return 0;
		if (o==p[0]) return 1;
		if (p.size()==1) return 0;
		ll tmp=(o-p[0])*(p.back()-p[0]);
		if (tmp==0) return dis2(o,p[0])<=dis2(p.back(),p[0]);
		if (tmp<0||p.size()==2) return 0;
		int x=upper_bound(1+all(p),o,[&](const point<db> &a,const point<db> &b){return (a-p[0])*(b-p[0])>eps;})-p.begin()-1;
		return (o-p[x])*(p[x+1]-p[x])<=0;
	}
	tmpl struct half_plane//默认左侧
	{
		point<T> o,d;
		operator half_plane<ll>() const {return {(point<ll>)o,(point<ll>)d,0};}
		operator half_plane<db>() const {return {(point<db>)o,(point<db>)d,0};}
		half_plane(){}
		half_plane(const point<T> &a,const point<T> &b,bool twopoint)
		{
			o=a;
			d=twopoint?b-a:b;
		}
		bool operator<(const half_plane<T> &a) const 
		{
			int p=quad(d),q=quad(a.d);
			if (p!=q) return p<q;
			p=sgn(d*a.d);
			if (p) return p>0;
			return sgn(d*(a.o-o))>0;
		}
	};
	tmpl ostream & operator<<(ostream &cout,half_plane<T> &m) {return cout<<m.o<<" | "<<m.d;}
	tmpl point<db> intersect(const half_plane<T> &m,const half_plane<T> &n)
	{
		if (!sgn(m.d*n.d))
		{
			if (!sgn(m.d*(n.o-m.o))) return apos;
			return npos;
		}
		return (point<db>)m.o+(n.o-m.o)*n.d/(db)(m.d*n.d)*(point<db>)m.d;
	}
	const db inf=1e9;
	tmpl convex<db> intersect(vector<half_plane<T>> a)
	{
		T I=inf;
		a.push_back({{-I,-I},{I,-I},1});
		a.push_back({{I,-I},{I,I},1});
		a.push_back({{I,I},{-I,I},1});
		a.push_back({{-I,I},{-I,-I},1});
		sort(all(a));
		int n=a.size(),i,h=0,t=-1;
		half_plane<db> q[n];
		point<db> p[n];
		vector<point<db>> r;
		for (i=0;i<n;i++) if (i==n-1||sgn(a[i].d*a[i+1].d))
		{
			auto x=(half_plane<db>)a[i];
			while (h<t&&sgn((p[t-1]-x.o)*x.d)>=0) --t;
			while (h<t&&sgn((p[h]-x.o)*x.d)>=0) ++h;
			q[++t]=x;
			if (h<t) p[t-1]=intersect(q[t-1],q[t]);
		}
		while (h<t&&sgn((p[t-1]-q[h].o)*q[h].d)>=0) --t;
		if (h==t) return convex<db>(vector<point<db>>(0));
		p[t]=intersect(q[h],q[t]);
		return convex<db>(vector<point<db>>(p+h,p+t+1));
	}
	#undef tmpl
}
using geometry::point,geometry::line,geometry::circle,geometry::convex,geometry::half_plane;
using geometry::db,geometry::sgn,geometry::eps,geometry::ll,geometry::segment;
using geometry::intersect;
point<ll> a[55];
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	cout<<setiosflags(ios::fixed)<<setprecision(3);
	vector<half_plane<ll>> v;
	int m;cin>>m;
	while (m--)
	{
		int n,i;
		cin>>n;
		for (i=0;i<n;i++) cin>>a[i];
		a[n]=a[0];
		for (i=0;i<n;i++) v.push_back({a[i],a[i+1],1});
	}
	cout<<intersect(v).area()<<endl;
}
```

<div style="page-break-after:always"></div>

### 公式与杂项

#### 枚举大小为 r 的集合

思路：通过进位创造 1，再把一串 1 移到最后

```cpp
for (int s=(1<<r)-1;s<1<<n;)
{
	int t=s+(s&-s);
	s=(s&~t)>>lg[s&-s]+1|t;
}
```

#### 整体二分（区间 $k$-th）

$O((n+q)\log a)$，$O(n+q)$。

```cpp
struct cz
{
	int x,y,kth,pos,typ;
};
cz q[M],st1[M],st2[M];
int a[N],b[N],d[N],ans[N],s[N];
int n,m,t1,t2,i,j,c,gs;
int lb(int x)
{
	return x&(-x);
}
void add(int x,int y)
{
	for (;x<=n;x+=lb(x)) s[x]+=y;
}
int sum(int x)
{
	int ans=0;
	for (;x;x-=lb(x)) ans+=s[x];
	return ans;
}
void ztef(int ql,int qr,int l,int r)
{
	if (ql>qr) return;
	int mid=l+r>>1,i,midd;
	t1=t2=0;
	if (l==r)
	{
		for (i=ql;i<=qr;i++) if (q[i].typ) ans[q[i].pos]=d[l];
		return;
	}
	for (i=ql;i<=qr;i++) if (q[i].typ)
	{
		midd=sum(q[i].y)-sum(q[i].x-1);
		if (midd>=q[i].kth) st1[++t1]=q[i]; else
		{
			st2[++t2]=q[i];
			st2[t2].kth-=midd;
		}
	}
	else if (q[i].pos<=mid)
	{
		add(q[i].x,1);
		st1[++t1]=q[i];
	}
	else st2[++t2]=q[i];
	for (i=1;i<=t1;i++) if (!st1[i].typ) add(st1[i].x,-1);
	for (i=1;i<=t1;i++) q[i+ql-1]=st1[i];
	midd=ql+t1-1;
	for (i=1;i<=t2;i++) q[i+midd]=st2[i];
	ztef(ql,midd,l,mid);ztef(midd+1,qr,mid+1,r);
}
int main()
{
	read(n);read(m);
	for (i=1;i<=n;i++)
	{
		read(a[i]);b[i]=a[i];
	}
	sort(b+1,b+n+1);
	d[gs=1]=b[1];
	for (i=2;i<=n;i++) if (b[i]!=b[i-1]) d[++gs]=b[i];
	for (i=1;i<=n;i++) a[i]=lower_bound(d+1,d+gs+1,a[i])-d;
	for (i=1;i<=n;i++)
	{
		q[i].x=i;q[i].pos=a[i];q[i].typ=0;
	}
	for (i=1;i<=m;i++)
	{
		read(q[i+n].x);read(q[i+n].y);read(q[i+n].kth);q[i+n].pos=i;q[i+n].typ=1;
	}
	ztef(1,n+m,1,gs);
	for (i=1;i<=m;i++) printf("%d\n",ans[i]);
}
```

#### cdq 分治（三维偏序）

$O(n\log ^2n)$，$O(n)$。

```cpp
int lb(int x)
{
	return x&(-x);
}
void add(int x,int y)
{
	for (;x<=mx;x+=lb(x)) a[x]+=y;
}
int sum(int x)
{
	int ans=0;
	for (;x;x^=lb(x)) ans+=a[x];
	return ans;
}
void gb(int l,int r)
{
	int i=l,m=l+r>>1,j=m+1,p=l;
	if (i<m) gb(i,m);
	if (j<r) gb(j,r);
	while ((i<=m)||(j<=r)) if ((j>r)||(i<=m)&&(q[i].x<=q[j].x))
	{
		if (!q[i].typ) add(q[i].y,1);
		qq[p++]=q[i++];
	}
	else
	{
		if (q[j].typ) ans[q[j].pos]+=q[j].typ*sum(q[j].y);
		qq[p++]=q[j++];
	}
	for (i=l;i<=m;i++) if (!q[i].typ) add(q[i].y,-1);
	for (i=l;i<=r;i++) q[i]=qq[i];
}
int main()
{
	read(n);read(m);
	for (i=1;i<=n;i++)
	{
		read(q[i].x);read(q[i].y);++q[i].y;
		yc[i]=q[i].y;
		if (q[i].y>mx) mx=q[i].y;
	}
	qs=ys=n;
	for (i=1;i<=m;i++)
	{
		read(x);read(y);read(z);read(j);
		q[++qs].x=x-1;q[qs].y=y;q[qs].pos=i;q[qs].typ=1;
		q[++qs].x=z;q[qs].y=y;q[qs].pos=i;q[qs].typ=-1;
		q[++qs].x=x-1;q[qs].y=j+1;q[qs].pos=i;q[qs].typ=-1;
		q[++qs].x=z;q[qs].y=j+1;q[qs].pos=i;q[qs].typ=1;
		if (j+1>mx) mx=j+1;
	}
	gb(1,qs);
	for (i=1;i<=m;i++) printf("%d\n",ans[i]);
}
```

#### $k$ 阶差分（ $[L,R]$ 加 $\tbinom{j-L+k}{k}$）

$O((n+q)k)$，$O(nk)$。

```cpp
int main()
{
	read(n);read(m);
	for (i=1;i<=n;i++) read(b[i]);
	C[0][0]=1;
	for (i=1;i<=n+100;i++)
	{
		C[i][0]=1;
		for (j=1;j<=min(i,100);j++)
		{
			C[i][j]=C[i-1][j-1]+C[i-1][j];
			if (C[i][j]>=p) C[i][j]-=p;
		}
	}
	while (m--)
	{
		read(x);read(y);read(z);
		++a[x][z];
		for (i=0;i<=z;i++)
		{
			a[y+1][z-i]-=C[y-x+i][i];
			if (a[y+1][z-i]<0) a[y+1][z-i]+=p;
		}
	}
	for (i=100;i>=0;i--) for (j=1;j<=n;j++)
	{
		a[j][i]+=a[j-1][i];
		if (a[j][i]>=p) a[j][i]-=p;
		a[j][i]+=a[j][i+1];
		if (a[j][i]>=p) a[j][i]-=p;
	}
	for (i=1;i<=n;i++) printf("%d ",(b[i]+a[i][0])%p);
}
```

#### 高精度

```cpp
#include <bits/stdc++.h>
using namespace std;
namespace unsigned_bigint
{
	const int p=10000,ws=4;
	struct Q
	{
		vector<int> a;
		Q(){a.clear();}
		void operator=(const int &nn)
		{
			int n=nn;
			while (n) a.push_back(n%p),n/=p;
		}
		Q operator+(const Q &o) const
		{
			Q r;r.a.resize(max(a.size(),o.a.size()));if (!r.a.size()) return r;//resize&size?
			int len=r.a.size()-1,lenn=min(a.size(),o.a.size());
			for (int i=0;i<lenn;i++) r.a[i]=a[i]+o.a[i];
			if (a.size()>o.a.size()) for (int i=lenn;i<=len;i++) r.a[i]=a[i]; else for (int i=lenn;i<=len;i++) r.a[i]=o.a[i]; 
			for (int i=0;i<len;i++) if (r.a[i]>=p) r.a[i]-=p,++r.a[i+1];
			if (r.a[len]>=p) r.a.push_back(r.a[len]/p),r.a[len]%=p;
			return r;
		}
		Q operator-(const Q &o) const
		{
			Q r;r.a.resize(a.size());
			int len=o.a.size();
			for (int i=0;i<len;i++) r.a[i]=a[i]-o.a[i];
			memcpy(&r.a[o.a.size()],&a[o.a.size()],a.size()-o.a.size()<<2);
			len=a.size();
			for (int i=0;i<len;i++) if (r.a[i]<0) r.a[i]+=p,--r.a[i+1];
			while (r.a.size()&&!r.a[r.a.size()-1]) r.a.pop_back();
			return r;
		}
		Q operator*(const Q &o) const
		{
			Q r;r.a.resize(a.size()+o.a.size());
			if (!r.a.size()) return r;
			int n=a.size(),m=o.a.size();
			for (int i=0;i<n;i++) for (int j=0;j<m;j++) r.a[i+j]+=a[i]*o.a[j];
			n=r.a.size()-1;
			for (int i=0;i<n;i++) r.a[i+1]+=r.a[i]/p,r.a[i]%=p;
			if (!r.a[n]) r.a.pop_back();
			return r;
		}
		Q operator+(const int &o) const {Q r;r=o; return (*this)+r;}
		Q operator-(const int &o) const {Q r;r=o;return (*this)-r;}
		Q operator*(const int &o) const {Q r;r=o;return (*this)*r;}
		template<typename C> void operator+=(C &o)
		{
			Q r;
			r=(*this)+o;
			(*this)=r;
		}
		template<typename C> void operator-=(C &o)
		{
			Q r;
			r=(*this)-o;
			(*this)=r;
		}
		template<typename C> void operator*=(C &o)
		{
			Q r;
			r=(*this)*o;
			(*this)=r;
		}
		bool operator<(const Q &o)
		{
			if (a.size()^o.a.size()||!a.size()) return a.size()<o.a.size();
			for (int i=a.size()-1;~i;i--) if (a[i]^o.a[i]) return a[i]<o.a[i];
			return 0;
		}
		bool operator!=(const Q &o)
		{
			if (a.size()^o.a.size()) return 1;int n=a.size();
			for (int i=0;i<n;i++) if (a[i]^o.a[i]) return 1;
			return 0;
		}
		bool operator!=(const int &o)
		{
			Q r;r=o;
			return (*this)!=r;
		}
		bool operator==(const Q &o)
		{
			return !((*this)!=o);
		}
		bool operator==(const int &o)
		{
			Q r;r=o;
			return (*this)==r;
		}
		bool operator>(const Q &o)
		{
			if (a.size()^o.a.size()||!a.size()) return a.size()>o.a.size();
			for (int i=a.size()-1;~i;i--) if (a[i]^o.a[i]) return a[i]>o.a[i];
			return 0;
		}
		Q operator/(const int &o)
		{
			Q r=(*this);
			if (!a.size()) return r;
			for (int i=a.size()-1;i;i--) r.a[i-1]+=r.a[i]%o*p,r.a[i]/=o;
			r.a[0]/=o;
			while (r.a.size()&&!r.a[r.a.size()-1]) r.a.pop_back();
			return r;
		}
		void operator/=(const int &o)
		{
			if (!a.size()) return;
			for (int i=a.size()-1;i;i--) a[i-1]+=a[i]%o*p,a[i]/=o;
			a[0]/=o;
			while (a.size()&&!a[a.size()-1]) a.pop_back();
		}
		int operator%(const int &o)
		{
			if (!a.size()) return 0;
			if (p%o==0) return a[0]%o;
			int r=0;
			for (int i=a.size()-1;~i;i--) r=(r*p+a[i])%o;
			return r;
		}
	};
	istream & operator>>(istream &cin,Q &o)
	{
		o.a.clear();
		int cnt=0,n=0,r;
		string s;
		cin>>s;
		reverse(s.begin(),s.end());
		for (char c:s)
		{
			if (cnt==0) o.a.push_back(0),r=1;
			o.a[o.a.size()-1]+=(c^'0')*r;r*=10;
			if (++cnt==ws) cnt=0;//这里也要改，是压位的位数
		}//printf("%d\n",(int)o.a.size());
		return cin;
	}
	ostream & operator<<(ostream &cout,const Q &o)
	{
		if (!o.a.size()) return cout<<0;
		cout<<o.a.back();
		if (o.a.size()==1) return;
		for (int i=o.a.size()-2;~i;i--) cout<<setfill('0')<<setw(ws)<<o.a[i];//注意这里也要改
		return cout;
	}
	Q gcd(Q a,Q b)
	{
		Q r;r=1;
		while (a%2==0&&b%2==0) a/=2,b/=2,r=r*2;
		while (a%2==0) a/=2;
		while (b%2==0) b/=2;
		if (b<a) swap(a,b);
		while (a.a.size())
		{
			b=(b-a)/2;
			while (b.a.size()&&b%2==0) b/=2;
			if (b<a) swap(a,b);
		}
		return r*b;
	}
}
```

#### 分散层叠算法（Fractional Cascading）

$O(n+q(k+\log n))$，$O(n)$。

给出 $k$ 个长度为 $n$ 的**有序数组**。

现在有 $q$ 个查询 : 给出数 $x$，分别求出每个数组中大于等于 $x$ 的最小的数(非严格后继)。

若后继不存在，则定义为 $0$。你需要**在线地**回答这些询问。

```cpp
int a[M][N],b[M][N<<1],c[M][N<<1][2],len[M],ans[M];
int n,m,qs,p,q,d,i,j,x,y,la;
int main()
{
	read(n);read(m);read(qs);read(d);
	for (j=1;j<=m;j++) for (i=0;i<n;i++) read(a[j][i]);
	for (j=1;j<=m;j++) a[j][n]=inf+j;++n;
	for (i=0;i<n;i++) b[m][i]=a[m][i],c[m][i][0]=i;
	len[m]=n;
	for (j=m-1;j;j--)
	{
		p=0,q=1;
		while (p<n&&q<len[j+1]) 
if (a[j][p]<b[j+1][q]) b[j][len[j]]=a[j][p],c[j][len[j]][0]=p++,c[j][len[j]++][1]=q;
		else b[j][len[j]]=b[j+1][q],c[j][len[j]][0]=p,c[j][len[j]++][1]=q,q+=2;
		while (p<n) b[j][len[j]]=a[j][p],c[j][len[j]][0]=p++,c[j][len[j]++][1]=q;
		while (q<len[j+1]) b[j][len[j]]=b[j+1][q],c[j][len[j]][0]=p,c[j][len[j]++][1]=q,q+=2;
	}
	for (int ii=1;ii<=qs;ii++)
	{
		read(x);x^=la;
		y=lower_bound(b[1],b[1]+len[1],x)-b[1];
		ans[1]=a[1][c[1][y][0]];y=c[1][y][1];//下标是c[1][y][0]
		for (j=2;j<=m;j++) 
		{
			if (y&&b[j][y-1]>=x) --y;
			ans[j]=a[j][c[j][y][0]];//下标是c[j][y][0]
			y=c[j][y][1];
		}
		la=0;
		for (i=1;i<=m;i++) la^=ans[i]>inf?0:ans[i];
		if (ii%d==0) printf("%d\n",la);
	}
}
```

#### 模意义真分数还原

$q\equiv \dfrac{x}{a}\pmod p$，$|a|\le A$。

```cpp
pair<int, int> approx(int p,int q,int A)
{
	int x=q,y=p,a=1,b=0;
	while (x>A)
	{
		swap(x,y);swap(a,b);
		a-=x/y*b;x%=y;
	}
	return make_pair(x,a);
}
```

#### IO 优化

```cpp
	c[fread(c+1,1,N,stdin)+1]=0;char *cc=c;
void read(int &x)
{
	char *c=cc;
	while ((*c<48)||(*c>57)) ++c;
	x=*(c++)^48;
	while ((*c>=48)&&(*c<=57)) x=x*10+(*(c++)^48);cc=c;
}
void read(int &x)
{
	char *c=cc;fh=1;
	while ((*c<48)||(*c>57)){if (*c=='-') {++c;fh=-1;break;}++c;}
	x=*(c++)^48;
	while ((*c>=48)&&(*c<=57)) x=x*10+(*(c++)^48);
	x*=fh;cc=c;
}
void write(const int x)
{
	while (x)
	{
		st[++tp]=x%10;
		x/=10;
	}
	char *c=nc;
	while (tp) *(++c)=st[tp--]|48;
	*(++c)=10;nc=c;
}
	char *nc=sc;
	fwrite(sc+1,1,stp,stdout);
```

#### 神秘的指令

```cpp
#pragma GCC optimize("Ofast")
#pragma GCC target("sse3","sse2","sse")
#pragma GCC target("avx","sse4","sse4.1","sse4.2","ssse3")
#pragma GCC target("f16c")
#pragma GCC target("fma","avx2")
#pragma GCC target("xop","fma4")
#pragma GCC optimize("inline","fast-math","unroll-loops","no-stack-protector")
#pragma GCC diagnostic error "-fwhole-program"
#pragma GCC diagnostic error "-fcse-skip-blocks"
#pragma GCC diagnostic error "-funsafe-loop-optimizations"
#pragma GCC diagnostic error "-std=c++14"
```

#### 质数，$\omega(n)$ 与 $d(n)$
| $n$ | $n$ 前第一个质数 | $n$ 后第一个质数 | $\max\{\omega (n)\}$ | $\max\{d(n)\}$ |
|:-:|:-:|:-:|:-:|:-:|
| $10^{1}$  | $10^{1}-3$   |$10^{1}+1$   | $2$  | $4$      |
| $10^{2}$  | $10^{2}-3$   |$10^{2}+1$   | $3$  | $12$     |
| $10^{3}$  | $10^{3}-3$   |$10^{3}+13$  | $4$  | $32$     |
| $10^{4}$  | $10^{4}-27$  |$10^{4}+7$   | $5$  | $64$     |
| $10^{5}$  | $10^{5}-9$   |$10^{5}+3$   | $6$  | $128$    |
| $10^{6}$  | $10^{6}-17$  |$10^{6}+3$   | $7$  | $240$    |
| $10^{7}$  | $10^{7}-9$   |$10^{7}+19$  | $8$  | $448$    |
| $10^{8}$  | $10^{8}-11$  |$10^{8}+7$   | $8$  | $768$    |
| $10^{9}$  | $10^{9}-63$ |$10^{9}+7$   | $9$  | $1344$   |
| $10^{10}$ | $10^{10}-33$ |$10^{10}+19$ | $10$ | $2304$   |
| $10^{11}$ | $10^{11}-23$ |$10^{11}+3$  | $10$ | $4032$   |
| $10^{12}$ | $10^{12}-11$ |$10^{12}+39$ | $11$ | $6720$   |
| $10^{13}$ | $10^{13}-29$ |$10^{13}+37$ | $12$ | $10752$  |
| $10^{14}$ | $10^{14}-27$ |$10^{14}+31$ | $12$ | $17280$  |
| $10^{15}$ | $10^{15}-11$ |$10^{15}+37$ | $13$ | $26880$  |
| $10^{16}$ | $10^{16}-63$ |$10^{16}+61$ | $13$ | $41472$  |
| $10^{17}$ | $10^{17}-3$  |$10^{17}+3$  | $14$ | $64512$  |
| $10^{18}$ | $10^{18}-11$ |$10^{18}+3$  | $15$ | $103680$ |
| $10^{19}$ | $10^{19}-39$ |$10^{19}+51$ | $16$ | $161280$ |

#### NTT 质数

|$p=r\times 2^k+1$|$r$|$k$|$g$（最小原根）|
|:-:|:-:|:-:|:-:|
|$3$|$1$|$1$|$2$|
|$5$|$1$|$2$|$2$|
|$17$|$1$|$4$|$3$|
|$97$|$3$|$5$|$5$|
|$193$|$3$|$6$|$5$|
|$257$|$1$|$8$|$3$|
|$7681$|$15$|$9$|$17$|
|$12289$|$3$|$12$|$11$|
|$40961$|$5$|$13$|$3$|
|$65537$|$1$|$16$|$3$|
|$786433$|$3$|$18$|$10$|
|$5767169$|$11$|$19$|$3$|
|$7340033$|$7$|$20$|$3$|
|$23068673$|$11$|$21$|$3$|
|$104857601$|$25$|$22$|$3$|
|$167772161$|$5$|$25$|$3$|
|$469762049$|$7$|$26$|$3$|
|$998244353$|$119$|$23$|$3$|
|$1004535809$|$479$|$21$|$3$|
|$2013265921$|$15$|$27$|$31$|
|$2281701377$|$17$|$27$|$3$|
|$3221225473$|$3$|$30$|$5$|
|$75161927681$|$35$|$31$|$3$|
|$77309411329$|$9$|$33$|$7$|
|$206158430209$|$3$|$36$|$22$|
|$2061584302081$|$15$|$37$|$7$|
|$2748779069441$|$5$|$39$|$3$|
|$6597069766657$|$3$|$41$|$5$|
|$39582418599937$|$9$|$42$|$5$|
|$79164837199873$|$9$|$43$|$5$|
|$263882790666241$|$15$|$44$|$7$|
|$1231453023109121$|$35$|$45$|$3$|
|$1337006139375617$|$19$|$46$|$3$|
|$3799912185593857$|$27$|$47$|$5$|
|$4222124650659841$|$15$|$48$|$19$|
|$7881299347898369$|$7$|$50$|$6$|
|$31525197391593473$|$7$|$52$|$3$|
|$180143985094819841$|$5$|$55$|$6$|
|$1945555039024054273$|$27$|$56$|$5$|
|$4179340454199820289$|$29$|$57$|$3$|

#### 公式杂项

$n$ 个点 $k$ 个连通块的生成树方案 $n^{k-2}\prod\limits_{i=1}^k siz_i$

杜教筛 $g(1)S(n)=\sum\limits_{i=1}^n(f*g)(i)-\sum\limits_{j=2}^ng(j)S(\lfloor\frac nj\rfloor)$

$(x,y)$ 曼哈顿距离 $\to$ $(x+y,x-y)$ 切比雪夫距离  
$(x,y)$ 切比雪夫距离 $\to$ $(\dfrac{x+y}{2},\dfrac{x-y}{2})$ 曼哈顿距离

错排数=$\lceil0.5+\frac{n!}{e}\rceil$

Kummer's Theorem: $\tbinom{n+m}{n}$ 含 $p~(p\in \text {prime})$ 的次数是 $n+m$ 在 $p$ 进制下的进位数

$\ln (1-x^V)=-\sum\limits_{i\ge1}\dfrac{x^{Vi}}{i}$

$x^{\bar n}=\sum\limits_i S_1(n,i)x^i$

$\begin{cases}x\equiv a_1\pmod {m_1}\\x\equiv a_2\pmod {m_2}\\\cdots\\x\equiv a_n\pmod {m_n}\end{cases}$ 

$m_i$ 为不同的质数。设 $M=\prod\limits_{i=1}^nm_i$，$t_i\times \dfrac {M}{m_i}\equiv 1\pmod {m_i}$，则 $x\equiv \sum\limits_{i=1}^na_it_i\dfrac {M}{m_i}$。

$V-E+F=2$，$S=n+\frac s2-1$。（$n$ 为内部，$s$ 为边上）

$\pi^{-1}$ 最小时 $\pi$ 最小，$\pi$ 最大等价于 $\pi^{-1}$ 最大？

五边形数 GF：$\frac{x(2x+1)}{(1-x)^3}$

五边形数：$\frac{3n^2-n}2$，广义含非正，逆为分拆数 GF（注意系数正负和 $n$ 取值奇偶性相同）

贝尔数（划分集合方案数）EGF：$\exp(e^x-1)$，$B_n=\sum\limits_{i=0}^n S_2(n,i)$，伯努利数 EGF：$\dfrac{x}{e^x-1}$

$S_1(i,m)$ EGF：$\dfrac{(\sum\limits_{i\ge 0}\dfrac{x^i}i)^m}{m!}$，$S_2(i,m)$ EGF：$\dfrac{(e^x-1)^m}{m!}$

多项式牛顿迭代：如果已知 $G(F(x))\equiv0\pmod{x^{2n}}$，$G(F_*(x))\equiv0\pmod {x^n}$，则有 $F(x)\equiv F_*(x)-\frac{G(F_*(x))}{G'(F_*(x))}\pmod{x^{2n}}$。求导时孤立的多项式视为常数。

$\int_0^1 t^a(1-t)^b\mathrm{d}t=\dfrac{a!b!}{(a+b+1)!}$，$\sum\limits_{i=0}^{n-1}i^{\underline{k}}=\dfrac{n^{\underline{k+1}}}{k+1}$

Burnside 引理：等价类数量为 $\sum\limits_{g\in G}\dfrac{X^g}{|G|}$，$X^g$ 表示 $g$ 变换下不动点的数量。

Polya 定理：染色方案数为 $\sum\limits_{g\in G}\dfrac{m^{c(g)}}{|G|}$，其中 $c(g)$ 表示 $g$ 变换下环的数量。

假设已经只保留了一个牛人酋长，其名字为 $A=a_1a_2\cdots a_l$。

假设王国旁边开了一座赌场，每单位时间（就称为“秒”吧）会有一个赌徒带着 $1$ 铜币进入赌场。

赌场规则很简单：支付 $x$ 铜币赌下一秒会唱出 $y$，如果猜对了就返还 $nx$ 铜币，否则钱就没了。

每个赌徒会如下行动：支付 $1$ 铜币赌下一秒会唱出 $a_1$，如果赌对了就支付得到的 $n$ 铜币赌下一秒会唱出 $a_2$，如果还对了就支付得到的 $n^2$ 铜币赌下一秒会唱出 $a_3$，等等，以此类推，最后支付 $n^{l-1}$ 铜币赌下一秒会唱出 $a_l$。

一旦连续唱出了 $a_1a_2\cdots a_l$，赌场老板就会认为自己亏大了而关门，并驱散所有赌徒。

那么关门前发生了什么呢？以 $A=\{1,4,1,5,1,1,4,1\},n=5$ 为例：

- 最后一位赌徒拿着 $5$ 铜币离开；
- 倒数第三位赌徒拿着 $5^3$ 铜币离开；
- 倒数第八位赌徒拿着 $5^8$ 铜币离开；
- 其他所有赌徒空手而归。

我们可以发现 $1,3$ 恰好是原序列的所有 border 的长度，而且对于其他的名字也有这样的规律。

这时候最神奇的一步来了：由于这个赌博游戏是公平的，因此赌场应该期望下不赚不赔，因此关门时期望来了 $5+5^3+5^8$ 个赌徒，因此期望需要 $5+5^3+5^8$ 单位时间唱出这个名字。

同理，即可知道对于一般的 $A$，答案为：

$$\sum\limits_{a_1a_2\cdots a_c=a_{l-c+1}a_{l-c+2}\cdots a_l} n^c$$

<div style="page-break-after:always"></div>

 ### stl 使用指南

#### bitset

```cpp
#include <bits/stdc++.h>
using namespace std;
bitset<10> f(12);
char s2[] = "10101";
bitset<10> g(s2);
string s = "100101";
bitset<10> h(s);
int main()
{
	for (int i=0;i<=9;i++) if (f[i]) printf("1"); else printf("0");puts("");
	for (int i=0;i<=9;i++) if (g[i]) printf("1"); else printf("0");puts("");
	for (int i=0;i<=9;i++) if (h[i]) printf("1"); else printf("0");puts("");
    foo.count();//1的个数
	foo.flip();//全部翻转
	foo.set();//变1
	foo.reset();//变0
	foo.to_string();
	foo.to_ulong();
	foo.to_ullong();
	foo._Find_first();
	foo._Find_next();
    //位运算：<< 变大，>> 变小
    __builtin_clz();//前导 0
    __builtin_ctz();//后面的 0
}
```

输出：

```plain
0011000000
1010100000
1010010000
```

#### pb_ds

```cpp
#pragma GCC optimize("Ofast")
#pragma GCC target("sse3","sse2","sse")
#pragma GCC target("avx","sse4","sse4.1","sse4.2","ssse3")
#pragma GCC target("f16c")
#pragma GCC target("fma","avx2")
#pragma GCC target("xop","fma4")
#pragma GCC optimize("inline","fast-math","unroll-loops","no-stack-protector")
#pragma GCC diagnostic error "-fwhole-program"
#pragma GCC diagnostic error "-fcse-skip-blocks"
#pragma GCC diagnostic error "-funsafe-loop-optimizations"
#pragma GCC diagnostic error "-std=c++14"
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp> //balanced tree
#include <ext/pb_ds/hash_policy.hpp> //hash table
#include <ext/pb_ds/priority_queue.hpp> //priority_queue
using namespace __gnu_pbds;
using namespace std;
inline char gc()
{
    static char buf[1048576], *p1, *p2;
    return p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, 1048576, stdin),
    p1 == p2) ? EOF : *p1++;
}
inline int read()
{
    char ch = gc(); int r = 0, w = 1;
    for (; ch < '0' || ch > '9'; ch = gc()) if (ch == '-') w = -1;
    for (; '0' <= ch && ch <= '9'; ch = gc()) r = r * 10 + (ch - '0');
    return r * w;
}
typedef tree<int,null_type,less<int>,rb_tree_tag,tree_order_statistics_node_update> rbtree;
cc_hash_table<string,int>mp1;//拉链法
gp_hash_table<string,int>mp2;//查探法
rbtree s1,s2;//注意是不可重的
//null_type无映射(低版本g++为null_mapped_type)
//less<int>从小到大排序
//插入t.insert();
//删除t.erase();
//求有多少个数比 k 小:t.order_of_key(k);
//求树中第 k+1 小:t.find_by_order(k);
//a.join(b) b并入a，前提是两棵树的 key 的取值范围不相交，b 会清空但迭代器没事，如不满足会抛出异常
//a.split(v,b) key 小于等于 v 的元素属于 a，其余的属于 b
//T.lower_bound(x)  >=x 的 min 的迭代器
//T.upper_bound(x)  >x 的 min 的迭代器
__gnu_pbds::priority_queue<int,greater<int>,pairing_heap_tag> pq;
//join(priority_queue &other)  //合并两个堆,other会被清空
//split(Pred prd,priority_queue &other)  //分离出两个堆
//modify(point_iterator it,const key)  //修改一个节点的值
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	std::mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());	cout<<setiosflags(ios::fixed)<<setprecision(15);
	rbtree::iterator it;
    uniform_real_distribution<> dis(1, 2)
	numeric_limits<int>::max();
	for (int i=1;i<=10;i++) s1.insert(i*2);
	//it=s2.lower_bound(35);
	for (auto u:s1) printf("%d\n",u);puts("");
	printf("%d\n",*s1.find_by_order(10));
	//printf("%d\n",*it);
}
```

<div style="page-break-after:always"></div>

### fr 的板子（补充）

#### MTT+exp

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef double db;
int read(){
	int res=0;
	char c=getchar(),f=1;
	while(c<48||c>57){if(c=='-')f=0;c=getchar();}
	while(c>=48&&c<=57)res=(res<<3)+(res<<1)+(c&15),c=getchar();
	return f?res:-res;
}

const int L=1<<19,mod=1e9+7;
const db pi2=3.141592653589793*2;
int inc(int x,int y){return x+y>=mod?x+y-mod:x+y;}
int dec(int x,int y){return x-y<0?x-y+mod:x-y;}
int mul(int x,int y){return (ll)x*y%mod;}
int qpow(int x,int y){
	int res=1;
	for(;y;y>>=1)res=y&1?mul(res,x):res,x=mul(x,x);
	return res;
}
int inv(int x){return qpow(x,mod-2);}

struct cp{
	db x,y;
	cp(){}
	cp(db a,db b){x=a,y=b;}
	cp operator+(const cp& p)const{return cp(x+p.x,y+p.y);}
	cp operator-(const cp& p)const{return cp(x-p.x,y-p.y);}
	cp operator*(const cp& p)const{return cp(x*p.x-y*p.y,x*p.y+y*p.x);}
	cp conj(){return cp(x,-y);}
}w[L];
int re[L];
int getre(int n){
	int len=1,bit=0;
	while(len<n)++bit,len<<=1;
	for(int i=1;i<len;++i)re[i]=(re[i>>1]>>1)|((i&1)<<(bit-1));
	return len;
}
void getw(){
	for(int i=0;i<L;++i)w[i]=cp(cos(pi2/L*i),sin(pi2/L*i));
}
void fft(cp* a,int len,int m){
	for(int i=1;i<len;++i)if(i<re[i])swap(a[i],a[re[i]]);
	for(int k=1,r=L>>1;k<len;k<<=1,r>>=1)
		for(int i=0;i<len;i+=k<<1)
			for(int j=0;j<k;++j){
				cp &L=a[i+j],&R=a[i+j+k],t=w[r*j]*R;
				R=L-t,L=L+t;
			}
	if(!~m){
		reverse(a+1,a+len);
		cp tmp=cp(1.0/len,0);
		for(int i=0;i<len;++i)a[i]=a[i]*tmp;
	}
}
void mul(int* a,int* b,int* c,int n1,int n2,int n){
	static cp f1[L],f2[L],f3[L],f4[L];
	int len=getre(n1+n2-1);
	for(int i=0;i<len;++i){
		f1[i]=i<n1?cp(a[i]>>15,a[i]&32767):cp(0,0);
		f2[i]=i<n2?cp(b[i]>>15,b[i]&32767):cp(0,0);
	}
	fft(f1,len,1),fft(f2,len,1);
	cp t1=cp(0.5,0),t2=cp(0,-0.5),r=cp(0,1);
	cp x1,x2,x3,x4;
	for(int i=0;i<len;++i){
		int j=(len-i)&(len-1);
		x1=(f1[i]+f1[j].conj())*t1;
		x2=(f1[i]-f1[j].conj())*t2;
		x3=(f2[i]+f2[j].conj())*t1;
		x4=(f2[i]-f2[j].conj())*t2;
		f3[i]=x1*(x3+x4*r);
		f4[i]=x2*(x3+x4*r);
	}
	fft(f3,len,-1),fft(f4,len,-1);
	ll c1,c2,c3,c4;
	for(int i=0;i<n;++i){
		c1=(ll)(f3[i].x+0.5)%mod,c2=(ll)(f3[i].y+0.5)%mod;
		c3=(ll)(f4[i].x+0.5)%mod,c4=(ll)(f4[i].y+0.5)%mod;
		c[i]=((((c1<<15)+c2+c3)<<15)+c4)%mod;
	}
}
void inv(int* a,int* b,int n){
	if(n==1){b[0]=1;return;}
	static int c[L];
	int l=(n+1)>>1;
	inv(a,b,l);
	mul(a,b,c,n,l,n);
	for(int i=0;i<n;++i)c[i]=mod-c[i];
	c[0]+=2;
	mul(b,c,b,n,n,n);
}
void der(int* a,int n){
	for(int i=1;i<n;++i)a[i-1]=mul(a[i],i);
	a[n-1]=0;
}
void its(int* a,int n){
	for(int i=n-1;i;--i)a[i]=mul(a[i-1],inv(i));
	a[0]=0;
}
void ln(int* a,int* b,int n){
	static int c[L];
	for(int i=0;i<n;++i)c[i]=a[i];
	der(c,n);
	inv(a,b,n);
	mul(b,c,b,n,n,n);
	its(b,n);
}
void exp(int* a,int* b,int n){
	if(n==1){b[0]=1;return;}
	static int c[L];
	int l=(n+1)>>1;
	exp(a,b,l);
	ln(b,c,n);
	for(int i=0;i<n;++i)c[i]=dec(a[i],c[i]);
	++c[0];
	mul(b,c,b,l,n,n);
	for(int i=0;i<n;++i)c[i]=0;
}

int n,k,a[L],f[L],g[L];
int main(){
	getw();
	n=read(),k=read();
	for(int i=1;i<=k;++i)a[i]=inv(i);
	for(int i=2;i<=n;++i)
		for(int j=1;i*j<=k;++j)
			f[i*j]=inc(f[i*j],a[j]);
	for(int i=1;i<=k;++i)f[i]=mod-f[i];
	for(int i=1;i<=k;++i)f[i]=inc(f[i],mul(n-1,a[i]));
	exp(f,g,k+1);
	printf("%d\n",g[k]);
}
```

#### 多项式

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
int read(){
	int res=0;
	char c=getchar(),f=1;
	while(c<48||c>57){if(c=='-')f=0;c=getchar();}
	while(c>=48&&c<=57)res=(res<<3)+(res<<1)+(c&15),c=getchar();
	return f?res:-res;
}
void write(int x){
	char c[21];
	int len=0;
	if(!x)return putchar('0'),void();
	if(x<0)x=-x,putchar('-');
	while(x)c[++len]=x%10,x/=10;
	while(len)putchar(c[len--]+48);
}
#define space(x) write(x),putchar(' ')
#define enter(x) write(x),putchar('\n')

const int mod=998244353;
struct M{
	int x;
	M(int a=0):x(a){}
	M operator+(const M& p)const{return x+p.x>=mod?x+p.x-mod:x+p.x;}
	M operator-()const{return x?mod-x:0;}
	M operator-(const M& p)const{return x-p.x<0?x-p.x+mod:x-p.x;}
	M operator*(const M& p)const{return (ll)x*p.x%mod;}
	bool operator==(const int& p)const{return x==p;}
	void operator+=(const M& p){*this=*this+p;}
	void operator-=(const M& p){*this=*this-p;}
	void operator*=(const M& p){*this=*this*p;}
};
void write(const M& x){write(x.x);}

M qpow(M x,int y){
	M res(1);
	for(;y;y>>=1)res=y&1?res*x:res,x=x*x;
	return res;
}
M inv(M x){return qpow(x,mod-2);}

const int N=1<<21|7;
namespace NTT{
int re[N];
M w[2][N];
int getre(int n){
	int len=1,bit=0;
	while(len<n)len<<=1,++bit;
	for(int i=1;i<len;++i)re[i]=(re[i>>1]>>1)|((i&1)<<(bit-1));
	w[0][0]=w[1][0]=1,w[0][1]=qpow(3,(mod-1)/len),w[1][1]=inv(w[0][1]);
	for(int o=0;o<2;++o)for(int i=2;i<=len;++i)
		w[o][i]=w[o][i-1]*w[o][1];
	return len;
}
void NTT(M* a,int n,int o=0){
	for(int i=1;i<n;++i)if(i<re[i])swap(a[i],a[re[i]]);
	M L,R;
	for(int k=1;k<n;k<<=1)
		for(int i=0,st=n/(k<<1);i<n;i+=k<<1)
			for(int j=0,nw=0;j<k;++j,nw+=st){
				L=a[i+j],R=a[i+j+k]*w[o][nw];
				a[i+j]=L+R,a[i+j+k]=L-R;
			}
	if(o){
		L=inv(n);
		for(int i=0;i<n;++i)a[i]=a[i]*L;
	}
}

M t0[N],t1[N],t2[N];
void mul(const M* a,const M* b,M* c,int n,int m){
	int len=getre(n+m+1);
	memset(t0,0,sizeof(int)*len),memcpy(t0,a,sizeof(int)*(n+1));
	memset(t1,0,sizeof(int)*len),memcpy(t1,b,sizeof(int)*(m+1));
	NTT(t0,len),NTT(t1,len);
	for(int i=0;i<len;++i)t0[i]=t0[i]*t1[i];
	NTT(t0,len,1);
	memcpy(c,t0,sizeof(int)*(n+m+1));
}
void inv(const M* a,M* b,int n){
	int len=1;
	while(len<=n)len<<=1;
	memset(t0,0,sizeof(int)*len),memcpy(t0,a,sizeof(int)*(n+1));
	memset(t1,0,sizeof(int)*(len<<1));
	memset(t2,0,sizeof(int)*(len<<1));
	t2[0]=inv(t0[0]);
	for(int k=1;k<=len;k<<=1){
		memcpy(t1,t0,sizeof(int)*k);
		getre(k<<1);
		NTT(t1,k<<1),NTT(t2,k<<1);
		for(int i=0;i<(k<<1);++i)t2[i]*=(-t1[i]*t2[i]+2);
		NTT(t2,k<<1,1);
		for(int i=k;i<(k<<1);++i)t2[i]=0;
	}
	memcpy(b,t2,sizeof(int)*(n+1));
}
} //namespace NTT

struct poly:public vector<M>{
	int time()const{return size()-1;}
	poly(int tim=0,int c=0){
		resize(tim+1);
		if(tim>=0)at(0)=c;
	}
	poly operator%(const int& n)const{
		poly r(*this);
		r.resize(n);
		return r;
	}
	poly operator%=(const int& n){
		resize(n);
		return *this;
	}
	poly operator+(const poly& p)const{
		int n=time(),m=p.time();
		poly r(*this);
		if(n<m)r.resize(m+1);
		for(int i=0;i<=m;++i)r[i]+=p[i];
		return r;
	}
	poly operator-(const poly& p)const{
		int n=time(),m=p.time();
		poly r(*this);
		if(n<m)r.resize(m+1);
		for(int i=0;i<=m;++i)r[i]-=p[i];
		return r;
	}
	poly operator*(const poly& p)const{
		poly r(time()+p.time());
		NTT::mul(&((*this)[0]),&p[0],&r[0],time(),p.time());
		return r;
	}
};

poly inv(const poly& a){
	poly r(a.time());
	NTT::inv(&a[0],&r[0],a.time());
	return r;
}

poly der(const poly& a){
	int n=a.time();
	poly r(n-1);
	for(int i=1;i<=n;++i)r[i-1]=a[i]*i;
	return r;
}
M _[N];
poly itr(const poly& a){
	int n=a.time();
	poly r(n+1);
	_[1]=1;
	for(int i=2;i<=n+1;++i)_[i]=_[mod%i]*(mod-mod/i);
	for(int i=0;i<=n;++i)r[i+1]=a[i]*_[i+1];
	return r;
}
poly ln(const poly& a){
	return itr(der(a)*inv(a)%a.time());
}

poly exp(const poly& a){
	poly r(0,1);
	int n=a.time(),k=1;
	while(r.time()<n)
		r%=k,r=r*(a%k-ln(r)+poly(0,1))%k,k<<=1;
	return r%(n+1);
}

void read(poly& a,int n=-1){
	if(!~n)n=a.time();
	else a.resize(n+1);
	for(int i=0;i<=n;++i)a[i]=read();
}
void write(const poly& a,int n=-1){
	if(!~n)n=a.time();
	else n=min(n,a.time());
	for(int i=0;i<n;++i)space(a[i]);
	enter(a[n]);
}
```

#### 计算 $f(c^k)$（$k\in [0,m)$）

```cpp
int n,m,c,a[N];
M u[N],v[N];
poly f,g;
int main(){
	n=read()-1,c=read(),m=read()-1;
	for(int i=0;i<=n;++i)a[i]=read();
	for(int i=0;i<=n+m;++i)u[i]=qpow(c,(ll)i*(i+1)/2%(mod-1));
	for(int i=0,p=max(n,m);i<=p;++i)v[i]=inv(u[i]);
	f.resize(n+1),g.resize(n+m+1);
	for(int i=0;i<=n;++i)f[n-i]=v[i]*a[i];
	for(int i=0;i<=n+m;++i)g[i]=u[i];
	f=f*g;
	for(int i=0;i<=m;++i)space(f[i+n]*v[i]);
}
```

#### Miller Rabin/Pollard Rho

1s：$200$ 组 $10^{18}$。

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef __int128 lll;
typedef pair<ll,int> pa;
ll ksm(ll x,ll y,const ll p)
{
	ll r=1;
	while (y)
	{
		if (y&1) r=(lll)r*x%p;
		x=(lll)x*x%p;y>>=1;
	}
	return r;
}
namespace miller
{
	const int p[7]={2,3,5,7,11,61,24251};
	ll s,t;
	bool test(ll n,int p)
	{
		if (p>=n) return 1;
		ll r=ksm(p,t,n),w;
		for (int j=0;j<s&&r!=1;j++)
		{
			w=(lll)r*r%n;
			if (w==1&&r!=n-1) return 0;
			r=w;
		}
		return r==1;
	}
	bool prime(ll n)
	{
		if (n<2||n==46'856'248'255'981ll) return 0;
		for (int i=0;i<7;++i) if (n%p[i]==0) return n==p[i];
		s=__builtin_ctz(n-1);t=n-1>>s;
		for (int i=0;i<7;++i) if (!test(n,p[i])) return 0;
		return 1;
	}
}
using miller::prime;
mt19937_64 rnd(chrono::steady_clock::now().time_since_epoch().count());
namespace rho
{
	void nxt(ll &x,ll &y,ll &p) {x=((lll)x*x+y)%p;}
	ll find(ll n,ll C)
	{
		ll l,r,d,p=1;
		l=rnd()%(n-2)+2,r=l;
		nxt(r,C,n);
		int cnt=0;
		while (l^r)
		{
			p=(lll)p*llabs(l-r)%n;
			if(!p) return gcd(n,llabs(l-r));
			++cnt;
			if (cnt==127)
			{
				cnt=0;
				d=gcd(llabs(l-r),n);
				if(d>1) return d;
			}
			nxt(l,C,n);nxt(r,C,n);nxt(r,C,n);
		}
		return gcd(n,p);
	}
	vector<pa> w;
	vector<ll> d;
	void dfs(ll n,int cnt)
	{
		if (n==1) return;
		if (prime(n)) return w.emplace_back(n,cnt),void();
		ll p=n,C=rnd()%(n-1)+1;
		while (p==1||p==n) p=find(n,C++);
		int r=1;n/=p;
		while (n%p==0) n/=p,++r;
		dfs(p,r*cnt);dfs(n,cnt);
	}
	vector<pa> getw(ll n)
	{
		w=vector<pa>(0);dfs(n,1);
		if (n==1) return w;
		sort(w.begin(),w.end());
		int i,j;
		for (i=1,j=0;i<w.size();i++) if (w[i].first==w[j].first) w[j].second+=w[i].second; else w[++j]=w[i];
		w.resize(j+1);
		return w;
	}
	void dfss(int x,ll n)
	{
		if (x==w.size()) return d.push_back(n),void();
		dfss(x+1,n);
		for (int i=1;i<=w[x].second;i++) dfss(x+1,n*=w[x].first);
	}
	vector<ll> getd(ll n)
	{
		getw(n);d=vector<ll>(0);dfss(0,1);
		sort(d.begin(),d.end());
		return d;
	}
}
int main()
{
	ios::sync_with_stdio(0);
	int t;
	cin>>t;
	while (t--)
	{
		ll x;cin>>x;
		auto v=rho::getw(x);
		int ans=0;
		for (auto &[u,v]:v) ans+=v;
		cout<<ans;
		for (auto &[u,v]:v) for (int i=1;i<=v;i++) cout<<' '<<u;cout<<'\n';
	}
}
```

#### 半平面交

```cpp
const int N=305;
const db inf=1e15,eps=1e-10;
int sign(db x){
	if(fabs(x)<eps)return 0;
	return x>0?1:-1;
}

struct vec{
	db x,y;
	vec(){}
	vec(db a,db b){x=a,y=b;}
	vec operator+(const vec& p)const{
		return vec(x+p.x,y+p.y);
	}
	vec operator-(const vec& p)const{
		return vec(x-p.x,y-p.y);
	}
	db operator*(const vec& p)const{
		return x*p.y-y*p.x;
	}
	vec operator*(const db& p)const{
		return vec(x*p,y*p);
	}
}p1[N],p2[N];

struct line{
	vec s,t;
	line(){}
	line(vec a,vec b){s=a,t=b;}
}a[N],q[N];
db ang(vec v){
	return atan2(v.y,v.x);
}
db ang(line l){
	return ang(l.t-l.s);
}
bool cmp(line x,line y){
	int s=sign(ang(x)-ang(y));
	return s?s<0:sign((x.t-x.s)*(y.t-x.s))>0;
}

vec inter(line x,line y){
	vec a=y.s-x.s,b=x.t-x.s,c=y.t-y.s;
	return y.s+c*((a*b)/(b*c));
}
bool out(line l,vec p){
	return sign((l.t-l.s)*(p-l.s))<0;
}

int n,tot=0;
db ans=inf;
int main(){
	scanf("%d",&n);
	for(int i=1;i<=n;++i)scanf("%lf",&p1[i].x);
	for(int i=1;i<=n;++i)scanf("%lf",&p1[i].y);
	for(int i=1;i<n;++i)a[i]=line(p1[i],p1[i+1]);
	a[n]=line(vec(p1[1].x,inf),vec(p1[1].x,p1[1].y));
	a[n+1]=line(vec(p1[n].x,p1[n].y),vec(p1[n].x,inf));
	
	sort(a+1,a+n+2,cmp);
	for(int i=1;i<=n;++i){
		if(!sign(ang(a[i])-ang(a[i+1])))continue;
		a[++tot]=a[i];
	}a[++tot]=a[n+1];
	
	int l=1,r=0;
	q[++r]=a[1],q[++r]=a[2];
	for(int i=3;i<=tot;++i){
		while(l<r&&out(a[i],inter(q[r],q[r-1])))--r;
		while(l<r&&out(a[i],inter(q[l],q[l+1])))++l;
		q[++r]=a[i];
	}
	while(l<r&&out(q[l],inter(q[r],q[r-1])))--r;
	while(l<r&&out(q[r],inter(q[l],q[l+1])))++l;
//......
}
```

#### 旋转卡壳

```cpp
if(top==3)return !printf("%d\n",dis(a[sta[1]],a[sta[2]]));
for(int i=1,j=2;i<top;++i){
	while(area(a[sta[i]],a[sta[i+1]],a[sta[j]])>=area(a[sta[i]],a[sta[i+1]],a[sta[j%top+1]]))j=j%top+1;
	ans=max(ans,max(dis(a[sta[i]],a[sta[j]]),dis(a[sta[i+1]],a[sta[j]])));
}printf("%d\n",ans);
```

#### l1ll5 trac

题意：

给一个n个点的完全图，边有颜色。将其扩展到(m+1)阶完全图，并且m染色，判断是否可行并输出方案。

题解：

Lemma 1：考虑最终所有 m(m+1)/2 条边，按照颜色分组，仅当每组都有(m+1)/2条边且m为偶数时有解

Proof：考虑边数最多的一组，令其有 x 条边，则其覆盖了 2x 个点，min(x) = (m+1)/2 ，当m为偶数时，2x=m，否则组内有重复点。

现在考虑n个点的情况，同样考虑最终的m组，将每组的(m+1)/2条边分为三类：

2-set：两个端点均已经在目前的图中出现

1-set：仅有一个端点在目前的图中出现（某点x的目前所有边中没有这个颜色，则在最终的图中必然有这种颜色的一条边）

0-set：以上两种情况之外的边，即完全没有出现。

Lemma 2：对于给出的情况，若没有某一组的1-set边数加2-set边数超过(m+1)/2，则有解。否则无解。

Proof：无解显然，为了证明有解，只需证明一个满足如上性质的图一定可以加一个点。（归纳，加一个点的影响是将边从1-set变成2-set（加某个颜色的边）或者从0-set变成1-set（不加某个颜色的边），不影响组内边数）

考虑用网络流解决该问题：对于每种颜色，将其和对应的1-set（n个点）或0-set（1个点）连边。其中0-set到T连一条m-n，其余都是1

显然源点总共发出m的流量，汇点最多收到m的流量。

不妨令每个颜色的点发出去的所有流量均相等，此时发现该网络满流。（不会证）

则最大流为满流，任取一整数最大流则对应一个方案。依次扩展到m+1个点即可。

感觉非常的玄妙。

因为l1ll5博客炸了，平时补的一些题就放在这里了。

CF 717 E

长度为n的排列，swap k次，能得到的不同排列数。

一个思路是考虑这个的逆过程，对所有swap j次能得到一个1-n的排列的排列计数，只需要考虑最后一个元素是否为错排即可

dp i j = dp i - 1, j + (i - 1) dp i - 1, j - 1

ans_i = dp[i][k] + dp[i - 2][k] + ... 这里考虑到浪费操作数就是交换相同对

但是复杂度是与n有关的，不妨考虑枚举做了k次swap，影响到了i个元素，答案乘一个C(n,i)即可

显然有重复，怎么避免呢，考虑只要枚举的i个元素都是错排即可不重不漏，容斥这个过程即可。

#### 多项式复合 (mrsrz)

$O(n\sqrt n\log n)$。

$1\le n\le 20000$。

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
#include<ctime>
const int N=65536,md=998244353;
typedef long long LL;
void upd(int&a){a+=a>>31&md;}
int pow(int a,int b){
    int ret=1;
    for(;b;b>>=1,a=(LL)a*a%md)if(b&1)ret=(LL)ret*a%md;
    return ret;
}
int n,m,A[N],B[N],L,g[17][N],lim,rev[N],M,B1[150][N],B2[250][N],R[N];
void init(int n){
    int l=-1;
    for(lim=1;lim<n;lim<<=1)++l;M=l+1;
    for(int i=1;i<lim;++i)
    rev[i]=((rev[i>>1])>>1)|((i&1)<<l);
}
void NTT(int*a,int f){
    for(int i=1;i<lim;++i)if(i<rev[i])std::swap(a[i],a[rev[i]]);
    for(int i=0;i<M;++i){
        const int*G=g[i],c=1<<i;
        for(int j=0;j<lim;j+=c<<1)
        for(int k=0;k<c;++k){
            const int x=a[j+k],y=a[j+k+c]*(LL)G[k]%md;
            upd(a[j+k]+=y-md),upd(a[j+k+c]=x-y);
        }
    }
    if(!f){
        const int iv=pow(lim,md-2);
        for(int i=0;i<lim;++i)a[i]=(LL)a[i]*iv%md;
        std::reverse(a+1,a+lim);
    }
}
void work(){
    B1[0][0]=B2[0][0]=1;
    for(int i=0;i<m;++i)B1[1][i]=B[i];
    NTT(B,1);
    for(int i=2;i<=L;++i){
        int*Bp=B1[i-1],*Bn=B1[i];
        NTT(Bp,1);
        for(int i=0;i<lim;++i)Bn[i]=(LL)B[i]*Bp[i]%md;
        NTT(Bp,0),NTT(Bn,0);
        for(int i=n;i<n<<1;++i)Bn[i]=0;
    }
    for(int i=0;i<n;++i)B2[1][i]=B1[L][i];
    int*bL=B1[L];
    NTT(bL,1);
    for(int i=2;i<L;++i){
        int*Bp=B2[i-1],*Bn=B2[i];
        NTT(Bp,1);
        for(int i=0;i<lim;++i)Bn[i]=(LL)bL[i]*Bp[i]%md;
        NTT(Bp,0),NTT(Bn,0);
        for(int i=n;i<n<<1;++i)Bn[i]=0;
    }
    NTT(bL,0);
}
int main(){
    scanf("%d%d",&n,&m),++n,++m;
    for(int i=0;i<17;++i){
        int*G=g[i];
        G[0]=1;
        const int gi=G[1]=pow(3,(md-1)/(1<<i+1));
        for(int j=2;j<1<<i;++j)G[j]=(LL)G[j-1]*gi%md;
    }
    L=sqrt(n)+1;
    init(n<<1);
    for(int i=0;i<n;++i)scanf("%d",A+i);
    for(int i=0;i<m;++i)scanf("%d",B+i);
    work();
    for(int i=0;i<L;++i){
        static int C[N];
        for(int i=0;i<lim;++i)C[i]=0;
        for(int j=0;j<L;++j){
            const int x=A[i*L+j],*B=B1[j];
            for(int k=0;k<n;++k)
            C[k]=(C[k]+(LL)x*B[k])%md;
        }
        NTT(C,1),NTT(B2[i],1);
        for(int j=0;j<lim;++j)C[j]=(LL)C[j]*B2[i][j]%md;
        NTT(C,0);
        for(int i=0;i<n;++i)upd(R[i]+=C[i]-md);
    }
    for(int i=0;i<n;++i)
    printf("%d ",R[i]);
    return 0;
}
```

#### 多项式复合 (yurzhang)

$O(n\log n\sqrt{n\log n})$，奇慢无比，慎用

```cpp
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
```

#### 下降幂多项式乘法

$O(n\log n)$。

```cpp
#include<cstdio>
#include<algorithm>
const int N=524288,md=998244353,g3=(md+1)/3;
typedef long long LL;
int n,m,A[N],B[N],fac[N],iv[N],rev[N],C[N],g[20][N],lim,M;
int pow(int a,int b){
    int ret=1;
    for(;b;b>>=1,a=(LL)a*a%md)if(b&1)ret=(LL)ret*a%md;
    return ret;
}
void upd(int&a){a+=a>>31&md;}
void init(int n){
    int l=-1;
    for(lim=1;lim<n;lim<<=1)++l;M=l+1;
    for(int i=1;i<lim;++i)
    rev[i]=((rev[i>>1])>>1)|((i&1)<<l);
}
void NTT(int*a,int f){
    for(int i=1;i<lim;++i)if(i<rev[i])std::swap(a[i],a[rev[i]]);
    for(int i=0;i<M;++i){
        const int*G=g[i],c=1<<i;
        for(int j=0;j<lim;j+=c<<1)
        for(int k=0;k<c;++k){
            const int x=a[j+k],y=a[j+k+c]*(LL)G[k]%md;
            upd(a[j+k]+=y-md),upd(a[j+k+c]=x-y);
        }
    }
    if(!f){
        const int iv=pow(lim,md-2);
        for(int i=0;i<lim;++i)a[i]=(LL)a[i]*iv%md;
        std::reverse(a+1,a+lim);
    }
}
int main(){
    scanf("%d%d",&n,&m);++n,++m;
    for(int i=0;i<20;++i){
        int*G=g[i];
        G[0]=1;
        const int gi=G[1]=pow(3,(md-1)/(1<<i+1));
        for(int j=2;j<1<<i;++j)G[j]=(LL)G[j-1]*gi%md;
    }
    for(int i=0;i<n;++i)scanf("%d",A+i);
    for(int i=0;i<m;++i)scanf("%d",B+i);
    for(int i=*fac=1;i<N;++i)
    fac[i]=fac[i-1]*(LL)i%md;
    iv[N-1]=pow(fac[N-1],md-2);
    for(int i=N-2;~i;--i)iv[i]=(i+1LL)*iv[i+1]%md;
    init(n+m<<1);
    for(int i=0;i<n+m-1;++i)C[i]=iv[i];
    NTT(A,1),NTT(B,1),NTT(C,1);
    for(int i=0;i<lim;++i)A[i]=(LL)A[i]*C[i]%md,B[i]=(LL)B[i]*C[i]%md;
    NTT(A,0),NTT(B,0);
    for(int i=0;i<lim;++i)C[i]=0;
    for(int i=0;i<n+m-1;++i)
    C[i]=(i&1)?md-iv[i]:iv[i];
    for(int i=0;i<lim;++i)A[i]=(LL)A[i]*B[i]%md*fac[i]%md;
    for(int i=n+m-1;i<lim;++i)A[i]=0;
    NTT(A,1),NTT(C,1);
    for(int i=0;i<lim;++i)A[i]=(LL)A[i]*C[i]%md;
    NTT(A,0);
    for(int i=0;i<n+m-1;++i)printf("%d%c",A[i]," \n"[i==n+m-2]);
    return 0;
}
```

#### 子集卷积

$O(n^22^n)$，$O(n2^n)$。

```cpp
#include <bits/stdc++.h>
#define ll long long
const int mod=1e9+9;
int a[22][1<<21],b[22][1<<21],h[22][1<<21],n,t;
int ctz(int x){return __builtin_popcount(x);}

void fwt(int *a,int n,int flag){
	for(int i=1;i<n;i<<=1)
	for(int len=i<<1,j=0;j<n;j+=len)
	for(int k=0;k<i;++k){
		if (flag==1)a[j+k+i]=(a[j+k]+a[j+k+i])%mod;
		else a[j+k+i]=(a[j+k+i]-a[j+k]+mod)%mod;
	}
}


int main(){
	scanf("%d",&n);
	int lim=n;n=1<<n;
	for(int i=0;i<n;++i)
		scanf("%d",&a[ctz(i)][i]);
	for(int i=0;i<n;++i)
		scanf("%d",&b[ctz(i)][i]);
	for(int i=0;i<=lim;++i){
		fwt(a[i],n,1);fwt(b[i],n,1);
	}for(int i=0;i<=lim;++i)
		for(int j=0;j<=i;++j)
			for(int k=0;k<n;++k)h[i][k]=(h[i][k]+(ll)a[j][k]*b[i-j][k]%mod)%mod;
	for(int i=0;i<=lim;++i)fwt(h[i],n,-1);
	for(int i=0;i<n;++i)printf("%d ",h[ctz(i)][i]);
	return 0;
}
```
