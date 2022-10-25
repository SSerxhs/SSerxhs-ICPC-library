//#define revsum
template<typename T> struct sgt_nt
{
	int L,R,n;
	vector<int> _ys;
	int *ys;
	T *a;
	vector<T> _s;
	T *s;
	#ifdef revsum
	vector<T> _rs;
	T *rs;
	#endif
	int z,y,fir;
	T dt;
private:
	void build(int x,int l,int r)
	{
		if (l==r)
		{
			ys[l]=x;
			s[x]=a[l];
			#ifdef revsum
			rs[x]=a[l];
			#endif
			return;
		}
		int c=x*2,m=l+r>>1;
		build(c,l,m);build(c+1,m+1,r);
		s[x]=s[c]+s[c+1];
		#ifdef revsum
		rs[x]=rs[c+1]+rs[c];
		#endif
	}
	void init(int n)
	{
		_ys.resize(n+1);ys=_ys.data();
		n<<=2;
		_s.resize(n);s=_s.data();
		#ifdef revsum
		_rs.resize(n);rs=_rs.data();
		#endif
	}
public: 
	sgt_nt(T b,int lt,int rt)
	{
		assert(lt<=rt);
		L=lt;R=rt;init(n=R-L+1);
		vector<T> _a(n,b);
		a=_a.data()-1;
		build(1,1,n);
	}
	sgt_nt(T *b,int lt,int rt)
	{
		assert(lt<=rt);
		L=lt;R=rt;init(n=R-L+1);
		a=b+L-1;
		build(1,1,n);
	}
	void modify(int p,T b)
	{
		assert(L<=p&&p<=R);
		//cerr<<"modify "<<p<<" to "<<b<<endl;
		p=ys[p-L+1];
		s[p]=b;
		#ifdef revsum
		rs[p]=b;
		#endif
		while (p>>=1)
		{
			s[p]=s[p*2]+s[p*2+1];
			#ifdef revsum
			rs[p]=rs[p*2+1]+rs[p*2];
			#endif
		}
	}
private: void ask_sum(int x,int l,int r)
	{
		if (z<=l&&r<=y)
		{
			dt=fir?s[x]:dt+s[x];
			fir=0;
			return;
		}
		int c=x*2,m=l+r>>1;
		if (z<=m) ask_sum(c,l,m);
		if (y>m) ask_sum(c+1,m+1,r);
	}
public: T ask_sum(int l,int r)
	{
		assert(L<=l&&l<=r&&r<=R);
		z=l-L+1;y=r-L+1;fir=1;
		ask_sum(1,1,n);
		//cerr<<"sum of ["<<l<<','<<r<<"] = "<<dt<<endl;
		return dt;
	}
	#ifdef revsum
	private: void ask_revsum(int x,int l,int r)
	{
		if (z<=l&&r<=y)
		{
			dt=fir?rs[x]:dt+rs[x];
			fir=0;
			return;
		}
		int c=x*2,m=l+r>>1;
		if (y>m) ask_revsum(c+1,m+1,r);
		if (z<=m) ask_revsum(c,l,m);
	}
public: T ask_revsum(int l,int r)
	{
		assert(L<=r&&r<=l&&l<=R);
		z=r-L+1;y=l-L+1;fir=1;
		ask_revsum(1,1,n);
		//cerr<<"revsum of ["<<l<<','<<r<<"] = "<<dt<<endl;
		return dt;
	}
	#endif
};
