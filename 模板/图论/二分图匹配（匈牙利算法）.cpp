vector<pair<int,int>> match(int n,int m,vector<pair<int,int>> eg)//[0,n],[0,m]
{
	++n;++m;
	vector<pair<int,int>> r;
	vector<vector<int>> e(m);
	vector<int> lk(n,-1),ed(n,-1);
	int cur,ans=0;
	for (auto [u,v]:eg) e[v].push_back(u);
	auto dfs=[&](auto dfs,int u) -> bool
	{
		for (int v:e[u]) if (ed[v]!=cur)
		{
			ed[v]=cur;
			if (lk[v]==-1||dfs(dfs,lk[v]))
			{
				lk[v]=u;
				return 1;
			}
		}
		return 0;
	};
	for (int i=0;i<m;i++) cur=i,ans+=dfs(dfs,i);
	r.reserve(ans);
	for (int i=0;i<n;i++) if (lk[i]>=0) r.push_back({i,lk[i]});
	return r;
}
