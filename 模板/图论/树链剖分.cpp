namespace HLD
{
	const int N=5e5+2;
	vector<int> e[N];
	int dfn[N],nfd[N],dep[N],f[N],siz[N],hc[N],top[N];
	int id;
	void dfs1(int u)
	{
		siz[u]=1;
		for (int v:e[u]) if (v!=f[u])
		{
			dep[v]=dep[f[v]=u]+1;
			dfs1(v);
			siz[u]+=siz[v];
			if (siz[v]>siz[hc[u]]) hc[u]=v;
		}
	}
	void dfs2(int u)
	{
		dfn[u]=++id;
		nfd[id]=u;
		if (hc[u])
		{
			top[hc[u]]=top[u];
			dfs2(hc[u]);
			for (int v:e[u]) if (v!=hc[u]&&v!=f[u]) dfs2(top[v]=v);
		}
	}
	int lca(int u,int v)
	{
		while (top[u]!=top[v])
		{
			if (dep[top[u]]<dep[top[v]]) swap(u,v);
			u=f[top[u]];
		}
		if (dep[u]>dep[v]) swap(u,v);
		return u;
	}
	int dis(int u,int v)
	{
		return dep[u]+dep[v]-(dep[lca(u,v)]<<1);
	}
	void init(int n)
	{
		for (int i=1;i<=n;i++)
		{
			e[i].clear();
			f[i]=hc[i]=0;
		}
		id=0;
	}
	void fun(int root)
	{
		dep[root]=1;dfs1(root);dfs2(top[root]=root);
	}
	vector<pair<int,int>> get_path(int u,int v)//u->v，注意可能出现 [r>l]（表示反过来走）
	{
		//cerr<<"path from "<<u<<" to "<<v<<": ";
		vector<pair<int,int>> v1,v2;
		while (top[u]!=top[v])
		{
			if (dep[top[u]]>dep[top[v]]) v1.push_back({dfn[u],dfn[top[u]]}),u=f[top[u]];
			else v2.push_back({dfn[top[v]],dfn[v]}),v=f[top[v]];
		}
		v1.reserve(v1.size()+v2.size()+1);
		v1.push_back({dfn[u],dfn[v]});
		reverse(v2.begin(),v2.end());
		for (auto v:v2) v1.push_back(v);
		//for (auto [x,y]:v1) cerr<<"["<<x<<','<<y<<"] ";cerr<<endl;
		return v1;
	}
}
using HLD::e,HLD::lca,HLD::dis,HLD::dfn,HLD::nfd,HLD::dep,HLD::f,HLD::siz,HLD::get_path;
using HLD::fun,HLD::init;//5e5
