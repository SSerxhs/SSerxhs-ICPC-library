#include "bits/stdc++.h"
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
