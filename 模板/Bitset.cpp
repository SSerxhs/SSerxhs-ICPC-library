struct Bitset
{
	typedef unsigned int ui;
	typedef unsigned long long ll;
#define all(x) (x).begin(),(x).end()
	const static ll B=-1llu;
	vector<ll> a;
	int n;
	Bitset() {}
	Bitset(int _n) :n(_n),a(_n+63>>6) {}
	bool operator[](int x) const { assert(x>=0&&x<n); return a[x>>6]>>(x&63)&1; }
	void set(int x,bool y) { assert(x>=0&&x<n); a[x>>6]=(a[x>>6]&(B^1<<(x&63)))|((ll)y<<(x&63)); }
	void set(int x) { assert(x>=0&&x<n); a[x>>6]|=1llu<<(x&63); }
	void set() { memset(a.data(),0xff,a.size()*sizeof a[0]); a.back()&=(1llu<<1+(n-1&63))-1; }
	void reset(int x) { a[x>>6]&=~(1llu<<(x&63)); }
	void reset() { memset(a.data(),0,a.size()*sizeof a[0]); }
	int count() const
	{
		int r=0;
		for (ll x:a) r+=__builtin_popcountll(x);
		return r;
	}
	Bitset &operator|=(const Bitset &o)
	{
		assert(n==o.n);
		for (int i=0; i<a.size(); i++) a[i]|=o.a[i];
		return *this;
	}
	Bitset operator|(Bitset o) { o|=*this; return o; }
	Bitset &operator&=(const Bitset &o)
	{
		assert(n==o.n);
		for (int i=0; i<a.size(); i++) a[i]&=o.a[i];
		return *this;
	}
	Bitset operator&(Bitset o) { o&=*this; return o; }
	Bitset &operator^=(const Bitset &o)
	{
		assert(n==o.n);
		for (int i=0; i<a.size(); i++) a[i]^=o.a[i];
		return *this;
	}
	Bitset operator^(Bitset o) { o^=*this; return o; }
	Bitset operator~() const
	{
		auto r=*this;
		for (ll &x:r.a) x=~x;
		return r;
	}
	Bitset &operator<<=(int x)
	{
		if (x>=n)
		{
			fill(all(a),0);
			return *this;
		}
		assert(x>=0);
		int y=x>>6;
		x&=63;
		for (int i=(int)a.size()-1; i>y; i--) a[i]=a[i-y]<<x|a[i-y-1]>>64-x;
		a[y]=a[0]<<x;
		memset(a.data(),0,y*sizeof a[0]);
		// fill_n(a.begin(),y,0);
		a.back()&=(1llu<<1+(n-1&63))-1;
		return *this;
	}
	Bitset operator<<(int x)
	{
		auto r=*this;
		r<<=x;
		return r;
	}
	Bitset &operator>>=(int x)
	{
		if (x>=n)
		{
			fill(all(a),0);
			return *this;
		}
		assert(x>=0);
		int y=x>>6,R=(int)a.size()-y-1;
		x&=63;
		for (int i=0; i<R; i++) a[i]=a[i+y]>>x|a[i+y+1]<<64-x;
		a[R]=a.back()>>x;
		memset(a.data()+R+1,y*sizeof a[0]);
		// fill(R+1+all(a),0);
		return *this;
	}
	Bitset operator>>(int x)
	{
		auto r=*this;
		r>>=x;
		return r;
	}
	void range_set(int l,int r)//[l,r) to 1
	{
		if (l>>6==r>>6)
		{
			a[l>>6]|=(1llu<<r-l)-1<<(l&63);
			return;
		}
		if (l&63)
		{
			a[l>>6]|=~((1llu<<(l&63))-1);//[l&63,64)
			l=(l>>6)+1<<6;
		}
		if (r&63)
		{
			a[r>>6]|=(1llu<<(r&63))-1;
			r=(r>>6)-1<<6;
		}
		memset(a.data()+(l>>6),0xff,(r-l>>6)*sizeof a[0]);
	}
	void range_reset(int l,int r)//[l,r) to 0
	{
		if (l>>6==r>>6)
		{
			a[l>>6]&=~((1llu<<r-l)-1<<(l&63));
			return;
		}
		if (l&63)
		{
			a[l>>6]&=(1llu<<(l&63))-1;//[l&63,64)
			l=(l>>6)+1<<6;
		}
		if (r&63)
		{
			a[r>>6]&=~((1llu<<(r&63))-1);
			r=(r>>6)-1<<6;
		}
		memset(a.data()+(l>>6),0,(r-l>>6)*sizeof a[0]);
	}
	void range_set(int l,int r,bool x)//[l,r)
	{
		if (x) range_set(l,r);
		else range_reset(l,r);
	}
};