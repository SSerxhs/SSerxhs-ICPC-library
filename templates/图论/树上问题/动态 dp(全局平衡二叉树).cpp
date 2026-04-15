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
