#define Plus
#define Times
//#define Cover
template<const unsigned int p> struct sgt_gm
{
	typedef unsigned int ui;
	typedef unsigned long long ll;
	int L,R;
	vector<int> _l,_r,_ys;
	int *l,*r,*ys;
	vector<ui> _a;
	ui *a;
	vector<ui> _s,_len;
	ui *s,*len;
	#ifdef Plus
	vector<ui> _plz;
	ui* plz;
	#endif
	#ifdef Times
	vector<ui> _clz;
	ui* clz;
	#endif
	#ifdef Cover
	vector<ui> _cover;
	vector<int> _flg;
	ui *cover;
	int *flg;
	#endif
	int z,y;
	ui dt;
private:
	void pushup(int x)
	{
		int c=x*2;
		if ((s[x]=s[c]+s[c+1])>=p) s[x]-=p;
	}
	#ifdef Plus
	void plus(int x,ui y)
	{
		#ifdef Cover
		if (flg[x]) {if ((cover[x]+=y)>=p) cover[x]-=p;goto cv;}
		#endif
		if ((plz[x]+=y)>=p) plz[x]-=p;
		cv:s[x]=(s[x]+(ll)y*len[x])%p;
	}
	#endif
	#ifdef Times
	void times(int x,ui y)
	{
		#ifdef Cover
		if (flg[x]) {cover[x]=(ll)cover[x]*y%p;goto cv;}
		#endif
		#ifdef Plus
		plz[x]=(ll)plz[x]*y%p;
		#endif
		clz[x]=(ll)clz[x]*y%p;
		cv:s[x]=(ll)s[x]*y%p;;
	}
	#endif
	#ifdef Cover
	void cov(int x,ui y)
	{
		flg[x]=1;
		cover[x]=y;
		#ifdef Plus
		plz[x]=0;
		#endif
		#ifdef Times
		clz[x]=1;
		#endif
		s[x]=(ll)y*len[x]%p;
	}
	#endif
	void pushdown(int x)
	{
		int c=x*2;
		#ifdef Cover
		if (flg[x])
		{
			cov(c,cover[x]);cov(c+1,cover[x]);
			flg[x]=0;
			return;
		}
		#endif
		#ifdef Times
		if (clz[x]!=1)
		{
			times(c,clz[x]);times(c+1,clz[x]);
			clz[x]=1;
		}
		#endif
		#ifdef Plus
		if (plz[x])
		{
			plus(c,plz[x]);plus(c+1,plz[x]);
			plz[x]=0;
		}
		#endif
	}
	void build(int x)
	{
		#ifdef Plus
		plz[x]=0;
		#endif
		#ifdef Times
		clz[x]=1;
		#endif
		#ifdef Cover
		flg[x]=0; 
		#endif
		len[x]=r[x]-l[x]+1;
		if (l[x]==r[x])
		{
			ys[l[x]]=x;
			s[x]=a[l[x]];
			return;
		}
		int c=x*2;
		l[c]=l[x];r[c]=l[x]+r[x]>>1;
		l[c+1]=r[c]+1;r[c+1]=r[x];
		build(c);build(c+1);
		pushup(x); 
	}
	void init(int n)
	{
		_a.resize(n+1);a=_a.data();
		_ys.resize(n+1);ys=_ys.data();
		n<<=2;
		_l.resize(n);l=_l.data();
		_r.resize(n);r=_r.data();
		_s.resize(n);s=_s.data();
		_len.resize(n);len=_len.data();
		#ifdef Plus
		_plz.resize(n);plz=_plz.data();
		#endif
		#ifdef Times
		_clz.resize(n);clz=_clz.data();
		#endif
		#ifdef Cover
		_cover.resize(n);cover=_cover.data();
		_flg.resize(n);flg=_flg.data();
		#endif
	}
public: 
	template<typename T> sgt_gm(T b,int lt,int rt)
	{
		assert(lt<=rt);
		b=(b%(int)p+(int)p)%p;
		L=lt;R=rt;init(R-L+1);
		fill_n(a+1,R-L+1,b);
		r[l[1]=1]=R-L+1;build(1);
	}
	template<typename T> sgt_gm(T *b,int lt,int rt)
	{
		assert(lt<=rt);
		L=lt;R=rt;init(R-L+1);
		for (int i=L;i<=R;i++) a[i-L+1]=(b[i]%(int)p+(int)p)%p;
		r[l[1]=1]=R-L+1;build(1);
	}
	#ifdef Plus
private: void modify_plus(int x)
	{
		if (z<=l[x]&&r[x]<=y)
		{
			plus(x,dt);
			return;
		}
		int c=x*2;
		pushdown(x);
		if (z<=r[c]) modify_plus(c);
		if (y>r[c]) modify_plus(c+1);
		pushup(x);
	}
public: void modify_plus(int l,int r,ui x)
	{
		//cerr<<"plus "<<x<<" to ["<<l<<','<<r<<']'<<endl;
		assert(L<=l&&l<=r&&r<=R);
		dt=x;z=l-L+1;y=r-L+1;
		modify_plus(1);
	}
	#endif
	#ifdef Times
private: void modify_times(int x)
	{
		if (z<=l[x]&&r[x]<=y)
		{
			times(x,dt);
			return;
		}
		int c=x*2;
		pushdown(x);
		if (z<=r[c]) modify_times(c);
		if (y>r[c]) modify_times(c+1);
		pushup(x);
	}
public: void modify_times(int l,int r,ui x)
	{
		//cerr<<"times "<<x<<" to ["<<l<<','<<r<<']'<<endl;
		assert(L<=l&&l<=r&&r<=R);
		dt=x;z=l-L+1;y=r-L+1;
		modify_times(1);
	}
	#endif
	#ifdef Cover
private: void modify_cover(int x)
	{
		if (z<=l[x]&&r[x]<=y)
		{
			cov(x,dt);
			return;
		}
		int c=x*2;
		pushdown(x);
		if (z<=r[c]) modify_cover(c);
		if (y>r[c]) modify_cover(c+1);
		pushup(x);
	}
public: void modify_cover(int l,int r,ui x)
	{
		//cerr<<"cover "<<x<<" to ["<<l<<','<<r<<']'<<endl;
		assert(L<=l&&l<=r&&r<=R);
		dt=x;z=l-L+1;y=r-L+1;
		modify_cover(1);
	}
	#endif
private: void ask_sum(int x)
	{
		if (z<=l[x]&&r[x]<=y)
		{
			if ((dt+=s[x])>=p) dt-=p;
			return;
		}
		pushdown(x);
		int c=x*2;
		if (z<=r[c]) ask_sum(c);
		if (y>r[c]) ask_sum(c+1);
	}
public: ui ask_sum(int l,int r)
	{
		assert(L<=l&&l<=r&&r<=R);
		z=l-L+1;y=r-L+1;dt=0;
		ask_sum(1);
		//cerr<<"sum of ["<<l<<','<<r<<"] = "<<dt<<endl;
		return dt;
	}
};
