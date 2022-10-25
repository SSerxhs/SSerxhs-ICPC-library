struct bit
{
	typedef unsigned long long ll;
	vector<ll> a,b;
	int n;
	bit(){}
	bit(int nn):n(nn),a(nn+1),b(nn+1){}
	template<typename T> bit(int nn,T *c):n(nn),a(nn+1),b(nn+1)
	{
		for (int i=1;i<=n;i++) a[i]=b[i];
		for (int i=1;i<=n;i++) if (i+(i&-i)<=n) a[i+(i&-i)]+=a[i];
	}
	void clear() {fill(all(a),0);fill(all(b),0);}
	void mdf(int l,int r,int dt)
	{
		int i;++r;
		ll j;
		for (i=l;i<=n;i+=i&-i) a[i]+=dt;j=(ll)dt*l;
		for (i=l;i<=n;i+=i&-i) b[i]+=j;
		for (i=r;i<=n;i+=i&-i) a[i]-=dt;j=(ll)dt*r;
		for (i=r;i<=n;i+=i&-i) b[i]-=j;
	}
	ll presum(int x)
	{
		ll r=0;
		for (int i=x;i;i-=i&-i) r+=a[i];r*=(x+1);
		for (int i=x;i;i-=i&-i) r-=b[i];
		return r+s[x];
	}
	ll sum(int l,int r)
	{
		return presum(r)-presum(l-1);
	}
};