#include <bits/stdc++.h>
using namespace std;
const int N=1e5+2;
int id[N],a[N];
int cnt,n,i;
void qs(int l,int r)
{
	a[id[l+r>>1]]=++cnt;
	int i=l,j=l+r>>1,mid=cnt;
	swap(id[l+r>>1],id[l]);
	if (l<r)qs(l+1,r); 
}
int main()
{
	n=1e5;
	printf("%d\n",n);
	for (i=1;i<=n;i++) id[i]=i;
	qs(1,n);
	for (i=1;i<=n;i++) printf("%d%c",a[i],i==n?10:32);
}
