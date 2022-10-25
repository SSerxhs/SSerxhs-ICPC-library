struct bit
{
	vector<ui> a;
	int n;
	bit(){}
	bit(int nn):n(nn),a(nn+1){}
	template<typename T> bit(int nn,T *b):n(nn),a(nn+1)
	{
		for (int i=1;i<=n;i++) a[i]=b[i];
		for (int i=1;i<=n;i++) if (i+(i&-i)<=n) a[i+(i&-i)]+=a[i];
	}
	void add(int x,ui y)
	{
		//cerr<<"add "<<x<<" by "<<y<<endl;
		assert(1<=x&&x<=n);
		if ((a[x]+=y)>=p) a[x]-=p;
		while ((x+=x&-x)<=n) if ((a[x]+=y)>=p) a[x]-=p; 
	}
	ui sum(int x)
	{
		//cerr<<"sum "<<x;
		assert(0<=x&&x<=n);
		ll r=a[x];
		while (x^=x&-x) r+=a[x];
		//cerr<<"= "<<r<<endl;
		return r%p;
	}
	ui sum(int x,int y)
	{
		return (sum(y)+p-sum(x-1))%p;
	}
};