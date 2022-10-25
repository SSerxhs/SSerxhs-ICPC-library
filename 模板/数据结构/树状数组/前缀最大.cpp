template<typename typC> struct bit
{
	vector<typC> a;
	int n;
	bit(){}
	bit(int nn,typC val):n(nn),a(nn+1,val){}
	template<typename T> bit(int nn,T *b):n(nn),a(nn+1)
	{
		for (int i=1;i<=n;i++) a[i]=b[i];
		for (int i=1;i<=n;i++) if (i+(i&-i)<=n) a[i+(i&-i)]=max(a[i+(i&-i),a[i]]);
	}
	//void clear(typC val) {fill(all(a),val);}?
	void update(int x,typC y)
	{
		//cerr<<"add "<<x<<" by "<<y<<endl;
		assert(1<=x&&x<=n);
		a[x]=max(a[x],y);
		while ((x+=x&-x)<=n) a[x]=max(a[x],y); 
	}
	typC sum(int x)
	{
		//cerr<<"sum "<<x;
		assert(0<=x&&x<=n);
		typC r=a[x];
		while (x^=x&-x) r=max(r,a[x]);
		//cerr<<"= "<<r<<endl;
		return r;
	}
};//to min：共四处修改