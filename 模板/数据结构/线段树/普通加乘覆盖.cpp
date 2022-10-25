#define Plus
//#define Times
//#define Cover
//#define Max
//#define Min
#define Sum
template<typename T> struct sgt
{
	int L,R;
	vector<int> _l,_r,_ys;
	int *l,*r,*ys;
	vector<T> _a;
	T *a;
	#ifdef Plus
	vector<T> _plz;
	T* plz;
	#endif
	#ifdef Times
	vector<T> _clz;
	T* clz;
	#endif
	#ifdef Cover
	vector<T> _cover;
	vector<int> _flg;
	T *cover;
	int *flg;
	#endif
	#ifdef Max
	vector<T> _mx;
	T *mx;
	#endif
	#ifdef Min
	vector<T> _mn;
	T *mn;
	#endif
	#ifdef Sum
	vector<T> _s,_len;
	T *s,*len;
	#endif
	int z,y;
	T dt;
private:
	void pushup(int x)
	{
		int c=x*2;
		#ifdef Max
		mx[x]=max(mx[c],mx[c+1]);
		#endif
		#ifdef Min
		mn[x]=min(mn[c],mn[c+1]);
		#endif
		#ifdef Sum
		s[x]=s[c]+s[c+1];
		#endif
	}
	#ifdef Plus
	void plus(int x,T y)
	{
		#ifdef Cover
		if (flg[x]) {cover[x]+=y;goto cv;}
		#endif
	plz[x]+=y;
	cv:;
		#ifdef Max
		mx[x]+=y;
		#endif
		#ifdef Min
		mn[x]+=y;
		#endif
		#ifdef Sum
		s[x]+=y*len[x];
		#endif
	}
	#endif
	#ifdef Times
	void times(int x,T y)
	{
		#ifdef Cover
		if (flg[x]) {cover[x]*=y;goto cv;}
		#endif
		#ifdef Plus
		plz[x]*=y;
		#endif
		clz[x]*=y;
	cv:;
		#ifdef Max
		mx[x]*=y;
		#endif
		#ifdef Min
		mn[x]*=y;
		#endif
		#ifdef Sum
		s[x]*=y;
		#endif
	#if defined(Max)&&defined(Min)
		if (y<0) swap(mx[x],mn[x]);
		#endif
	}
	#endif
	#ifdef Cover
	void cov(int x,T y)
	{
		flg[x]=1;
		cover[x]=y;
		#ifdef Plus
		plz[x]=0;
		#endif
		#ifdef Times
		clz[x]=1;
		#endif
		#ifdef Max
		mx[x]=y;
		#endif
		#ifdef Min
		mn[x]=y;
		#endif
		#ifdef Sum
		s[x]=y*len[x];
		#endif
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
		#ifdef Sum
		len[x]=r[x]-l[x]+1;
		#endif
		if (l[x]==r[x])
		{
			ys[l[x]]=x;
			#ifdef Max
			mx[x]=a[l[x]];
			#endif
			#ifdef Min
			mn[x]=a[l[x]];
			#endif
			#ifdef Sum
			s[x]=a[l[x]];
			#endif
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
		#ifdef Max
		_mx.resize(n);mx=_mx.data();
		#endif
		#ifdef Min
		_mn.resize(n);mn=_mn.data();
		#endif
		#ifdef Sum
		_s.resize(n);s=_s.data();
		_len.resize(n);len=_len.data();
		#endif
	}
public: 
	template<typename TT> sgt(TT b,int lt,int rt)
	{
		assert(lt<=rt);
		L=lt;R=rt;init(R-L+1);
		fill_n(a+1,R-L+1,b);
		r[l[1]=1]=R-L+1;build(1);
	}
	template<typename TT> sgt(TT *b,int lt,int rt)
	{
		assert(lt<=rt);
		L=lt;R=rt;init(R-L+1);
		for (int i=L;i<=R;i++) a[i-L+1]=b[i];
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
public: void modify_plus(int l,int r,T x)
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
public: void modify_times(int l,int r,T x)
	{
		//cerr<<"times "<<x<<" to ["<<l<<','<<r<<']'<<endl;
		assert(L<=l&&l<=r&&r<=R);
		dt=x;z=l-L+1;y=r-L+1;
	#if defined(Max)^defined(Min)
		assert(x>=0);
		#endif
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
public: void modify_cover(int l,int r,T x)
	{
		//cerr<<"cover "<<x<<" to ["<<l<<','<<r<<']'<<endl;
		assert(L<=l&&l<=r&&r<=R);
		dt=x;z=l-L+1;y=r-L+1;
		modify_cover(1);
	}
	#endif
	#ifdef Max
private: void ask_max(int x)
	{
		if (z<=l[x]&&r[x]<=y)
		{
			dt=max(dt,mx[x]);
			return;
		}
		pushdown(x);
		int c=x*2;
		if (z<=r[c]) ask_max(c);
		if (y>r[c]) ask_max(c+1);
	}
public: T ask_max(int l,int r)
	{
		assert(L<=l&&l<=r&&r<=R);
		z=l-L+1;y=r-L+1;dt=numeric_limits<T>::min();
		ask_max(1);
		//cerr<<"max of ["<<l<<','<<r<<"] = "<<dt<<endl;
		return dt;
	}
	#endif
	#ifdef Min
private: void ask_min(int x)
	{
		if (z<=l[x]&&r[x]<=y)
		{
			dt=min(dt,mn[x]);
			return;
		}
		pushdown(x);
		int c=x*2;
		if (z<=r[c]) ask_min(c);
		if (y>r[c]) ask_min(c+1);
	}
public: T ask_min(int l,int r)
	{
		assert(L<=l&&l<=r&&r<=R);
		z=l-L+1;y=r-L+1;dt=numeric_limits<T>::max();
		ask_min(1);
		//cerr<<"min of ["<<l<<','<<r<<"] = "<<dt<<endl;
		return dt;
	}
	#endif
	#ifdef Sum
private: void ask_sum(int x)
	{
		if (z<=l[x]&&r[x]<=y)
		{
			dt+=s[x];
			return;
		}
		pushdown(x);
		int c=x*2;
		if (z<=r[c]) ask_sum(c);
		if (y>r[c]) ask_sum(c+1);
	}
public: T ask_sum(int l,int r)
	{
		assert(L<=l&&l<=r&&r<=R);
		z=l-L+1;y=r-L+1;dt=0;
		ask_sum(1);
		//cerr<<"sum of ["<<l<<','<<r<<"] = "<<dt<<endl;
		return dt;
	}
	#endif
};
