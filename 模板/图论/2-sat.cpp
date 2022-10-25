struct sat
{
	vector<vector<int>> e;
	vector<int> dfn,low,st,f,ed;
	int fs,tp,id,n;
	sat(int n):n(n),e(n*2),dfn(n*2,-1),low(n*2),st(n*2),f(n*2,-1),ed(n*2),fs(0),tp(-1),id(0){}
	void dfs(int u)
	{
		dfn[u]=low[u]=id++;
		ed[u]=1;st[++tp]=u;
		for (int v:e[u]) if (dfn[v]!=-1)
		{
			if (ed[v]) low[u]=min(low[u],dfn[v]);
		} else dfs(v),low[u]=min(low[u],low[v]);
		if (dfn[u]==low[u])
		{
			do
			{
				f[st[tp]]=fs;
				ed[st[tp]]=0;
			} while (st[tp--]!=u);
			++fs;
		}
	}
	void add(int u,bool x,int v,bool y)//d:dif
	{
		assert(u>=0&&u<n&&v>=0&&v<n);
		e[u+x*n].push_back(v+y*n);
		e[v+(y^1)*n].push_back(u+(x^1)*n);
	}
	void set(int u,bool x)
	{
		assert(u>=0&&u<n);
		e[u+(x^1)*n].push_back(u+x*n);
	}
	vector<int> getans()
	{
		int i;
		for (i=0;i<n*2;i++) if (dfn[i]==-1) dfs(i);
		vector<int> r(n);
		for (i=0;i<n;i++)
		{
			if (f[i]==f[i+n]) return {};
			r[i]=f[i]>f[i+n];
		}
		return r;
	}
};