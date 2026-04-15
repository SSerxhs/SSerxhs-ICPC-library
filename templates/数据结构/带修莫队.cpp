#include "bits/stdc++.h"
using namespace std;
typedef long long ll;
#define all(x) (x).begin(),(x).end()
const int N=1.4e5,M=1e6+2;
int a[N],ans[N],bel[N],cnt[M],sum,z,y,cur;
struct P
{
	int p,v;
};
struct Q
{
	int l,r,t,p;
	bool operator<(const Q &o) const
	{
		if (bel[l]!=bel[o.l]) return bel[l]<bel[o.l];
		if (bel[r]!=bel[o.r]) return (bel[l]&1)^bel[r]<bel[o.r];
		return (bel[r]&1)?t<o.t:t>o.t;
	}
};
Q b[N];
P d[N];
void add(const int &x) {sum+=!(cnt[a[x]]++);}
void del(const int &x) {sum-=!(--cnt[a[x]]);}
void mdf(const int &x)
{
	auto &[p,v]=d[x];
	if (z<=p&&p<=y) del(p);
	swap(a[p],v);
	if (z<=p&&p<=y) add(p);
}
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	int n,m,q1=0,q2=0,i,ksiz;
	cin>>n>>m;
	for (i=1;i<=n;i++) cin>>a[i];
	for (i=1;i<=m;i++)
	{
		char c;
		int l,r;
		cin>>c>>l>>r;
		if (c=='Q') ++q1,b[q1]={l,r,q2,q1};
		else d[++q2]={l,r};
	}
	ksiz=max(1.0,round(cbrt((ll)n*n)));
	for (i=1;i<=n;i++) bel[i]=i/ksiz;
	sort(b+1,b+q1+1);
	z=b[1].l;y=z-1;cur=0;
	for (i=1;i<=q1;i++)
	{
		auto [l,r,t,p]=b[i];
		while (z>l) add(--z);
		while (y<r) add(++y);
		while (z<l) del(z++);
		while (y>r) del(y--);
		while (cur<t) mdf(++cur);
		while (cur>t) mdf(cur--);
		ans[p]=sum;
	}
	for (i=1;i<=q1;i++) cout<<ans[i]<<'\n';
}

