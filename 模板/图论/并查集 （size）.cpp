struct union_set
{
	vector<int> f,siz;
	int n;
	union_set(){}
	union_set(int nn):n(nn),f(nn+1),siz(nn+1)
	{
		iota(all(f),0);
		fill(all(siz),1);
	}
	int getf(int u) {return f[u]==u?u:f[u]=getf(f[u]);}
	void merge(int u,int v)
	{
		u=getf(u);v=getf(v);
		if (u!=v) siz[v]+=siz[u];
		f[u]=v;
	}
	bool connected(int u,int v) {return getf(u)==getf(v);}
};