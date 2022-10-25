template<typename typC> struct bit
{
	vector<typC> a;
	int n;
	bit() {}
	bit(int nn):n(nn),a(nn+2) {}
	template<typename T> bit(int nn,T *b):n(nn),a(nn+1)
	{
		cerr<<"没测试你也敢用？"<<endl;
		for (int i=1; i<=n; i++) a[i]=b[i];
		for (int i=n; i; i--) if (i^(i&-i)) a[i^(i&-i)]+=a[i];
	}
	void clear() { fill(all(a),0); }
	void add(int x,typC y)
	{
		//cerr<<"add "<<x<<" by "<<y<<endl;
		assert(1<=x&&x<=n);
		a[x]+=y;
		while (x^=x&-x) a[x]+=y;
	}
	typC sum(int x)
	{
		//cerr<<"sum "<<x;
		assert(1<=x&&x<=n+1);
		typC r=a[x];
		while ((x+=x&-x)<=n) r+=a[x];
		//cerr<<"= "<<r<<endl;
		return r;
	}
	typC sum(int x,int y)
	{
		return sum(x)-sum(y+1);
	}
};