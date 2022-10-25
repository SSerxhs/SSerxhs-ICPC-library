struct union_set
{
	vector<int> f;
	int n;
	union_set(){}
	union_set(int nn):n(nn),f(nn+1)
	{
		iota(all(f),0);
	}
	int getf(int u) {return f[u]==u?u:f[u]=getf(f[u]);}
	void merge(int u,int v)
	{
		u=getf(u);v=getf(v);
		f[u]=v;
	}
	bool connected(int u,int v) {return getf(u)==getf(v);}
};