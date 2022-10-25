template<typename typC> struct bit
{
	vector<typC> a;
	int n;
	bit() {}
	bit(int nn):n(nn),a(nn+1) {}
	template<typename T> bit(int nn,T *b):n(nn),a(nn+1)
	{
		for (int i=1; i<=n; i++) a[i]=b[i];
		for (int i=1; i<=n; i++) if (i+(i&-i)<=n) a[i+(i&-i)]+=a[i];
	}
	void add(int x,typC y)
	{
		//cerr<<"add "<<x<<" by "<<y<<endl;
		assert(1<=x&&x<=n);
		a[x]+=y;
		while ((x+=x&-x)<=n) a[x]+=y;
	}
	typC sum(int x)
	{
		//cerr<<"sum "<<x;
		assert(0<=x&&x<=n);
		typC r=a[x];
		while (x^=x&-x) r+=a[x];
		//cerr<<"= "<<r<<endl;
		return r;
	}
	typC sum(int x,int y)
	{
		return sum(y)-sum(x-1);
	}
};