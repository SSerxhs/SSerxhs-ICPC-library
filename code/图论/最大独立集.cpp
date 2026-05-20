vector<int> indep_set(int n,const vector<pair<int,int>> &edges)//[0,n)
{
	vector<vector<int>> e(n);
	mt19937 rnd(998);
	vector<int> p(n),q(n),ed(n);
	iota(all(p),0);
	shuffle(all(p),rnd);
	for (int i=0;i<n;i++) q[p[i]]=i;
	for (auto [u,v]:edges)
	{
		e[p[u]].push_back(p[v]);
		e[p[v]].push_back(p[u]);
	}
	vector<int> r,cur;
	function<void(int)> dfs=[&](int u)
	{
		if (cur.size()+n-u<=r.size()) return;
		if (u==n)
		{
			r=cur;
			return;
		}
		if (!ed[u])
		{
			cur.push_back(u);
			for (int v:e[u]) ++ed[v];
			dfs(u+1);
			for (int v:e[u]) --ed[v];
			cur.pop_back();
		}
		if (ed[u]||e[u].size()) dfs(u+1);
	};dfs(0);
	for (int &x:r) x=q[x];
	sort(all(r));
	return r;
}
