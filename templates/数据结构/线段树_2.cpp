template<class info, class tag> struct sgt
{
	int n, shift;
	info *a;
	info tmp;
	vector<info> s;
	vector<tag> tg;
	vector<int> lz;
	bool flg;
	void build(int x, int l, int r)
	{
		if (l == r)
		{
			s[x] = (flg ? tmp : a[l]);
			return;
		}
		int c = x * 2, m = l + r >> 1;
		build(c, l, m); build(c + 1, m + 1, r);
		s[x] = s[c] + s[c + 1];
	}
	sgt(info *b, int L, int R) :n(R - L + 1), shift(L - 1), a(b + L - 1), s(R - L + 1 << 2), tg(R - L + 1 << 2), lz(R - L + 1 << 2)
	{
		flg = 0;
		build(1, 1, n);
	}//[L,R]
	sgt(info b, int L, int R) :n(R - L + 1), shift(L - 1), s(R - L + 1 << 2), tg(R - L + 1 << 2), lz(R - L + 1 << 2)
	{
		tmp = b;
		flg = 1;
		build(1, 1, n);
	}//[L,R]
	int z, y;
	info res;
	tag dt;
	bool fir;
private:
	void _modify(int x, int l, int r)
	{
		if (z <= l && r <= y)
		{
			s[x] += dt;
			if (lz[x]) tg[x] += dt; else tg[x] = dt;
			lz[x] = 1;
			return;
		}
		int c = x * 2, m = l + r >> 1;
		if (lz[x])
		{
			if (lz[c]) tg[c] += tg[x]; else tg[c] = tg[x];
			lz[c] = 1; s[c] += tg[x]; c ^= 1;
			if (lz[c]) tg[c] += tg[x]; else tg[c] = tg[x];
			lz[c] = 1; s[c] += tg[x]; c ^= 1;
			lz[x] = 0;
		}
		if (z <= m) _modify(c, l, m);
		if (m < y) _modify(c + 1, m + 1, r);
		s[x] = s[c] + s[c + 1];
	}
	void ask(int x, int l, int r)
	{
		if (z <= l && r <= y)
		{
			res = fir ? s[x] : res + s[x];
			fir = 0;
			return;
		}
		int c = x * 2, m = l + r >> 1;
		if (lz[x])
		{
			if (lz[c]) tg[c] += tg[x]; else tg[c] = tg[x];
			lz[c] = 1; s[c] += tg[x]; c ^= 1;
			if (lz[c]) tg[c] += tg[x]; else tg[c] = tg[x];
			lz[c] = 1; s[c] += tg[x]; c ^= 1;
			lz[x] = 0;
		}
		if (z <= m) ask(c, l, m);
		if (m < y) ask(c + 1, m + 1, r);
	}
	function<bool(info)> check;
	void find_left_most(int x, int l, int r)
	{
		if (r < z || z <= l && !check(s[x])) return;
		if (l == r) { y = l; res = s[x]; return; }
		int c = x * 2, m = l + r >> 1;
		if (lz[x])
		{
			if (lz[c]) tg[c] += tg[x]; else tg[c] = tg[x];
			lz[c] = 1; s[c] += tg[x]; c ^= 1;
			if (lz[c]) tg[c] += tg[x]; else tg[c] = tg[x];
			lz[c] = 1; s[c] += tg[x]; c ^= 1;
			lz[x] = 0;
		}
		find_left_most(c, l, m);
		if (y == n + 1) find_left_most(c + 1, m + 1, r);
	}
	void find_right_most(int x, int l, int r)
	{
		if (l > y || r <= y && !check(s[x])) return;
		if (l == r) { z = l; res = s[x]; return; }
		int c = x * 2, m = l + r >> 1;
		if (lz[x])
		{
			if (lz[c]) tg[c] += tg[x]; else tg[c] = tg[x];
			lz[c] = 1; s[c] += tg[x]; c ^= 1;
			if (lz[c]) tg[c] += tg[x]; else tg[c] = tg[x];
			lz[c] = 1; s[c] += tg[x]; c ^= 1;
			lz[x] = 0;
		}
		find_right_most(c + 1, m + 1, r);
		if (z == 0) find_right_most(c, l, m);
	}
public:
	void modify(int l, int r, const tag &x)//[l,r]
	{
		z = l - shift; y = r - shift; dt = x;
		// cerr<<"modify ["<<l<<','<<r<<"] "<<'\n';
		assert(1 <= z && z <= y && y <= n);
		_modify(1, 1, n);
	}
	void modify(int pos, const info &o)
	{
		pos -= shift;
		int l = 1, r = n, m, c, x = 1;
		while (l < r)
		{
			c = x * 2; m = l + r >> 1;
			if (lz[x])
			{
				if (lz[c]) tg[c] += tg[x]; else tg[c] = tg[x];
				lz[c] = 1; s[c] += tg[x]; c ^= 1;
				if (lz[c]) tg[c] += tg[x]; else tg[c] = tg[x];
				lz[c] = 1; s[c] += tg[x]; c ^= 1;
				lz[x] = 0;
			}
			if (pos <= m) x = c, r = m; else x = c + 1, l = m + 1;
		}
		s[x] = o;
		while (x >>= 1) s[x] = s[x * 2] + s[x * 2 + 1];
	}
	info ask(int l, int r)//[l,r]
	{
		z = l - shift; y = r - shift; fir = 1;
		// cerr<<"ask ["<<l<<','<<r<<"] "<<'\n';
		assert(1 <= z && z <= y && y <= n);
		ask(1, 1, n);
		return res;
	}
	pair<int, info> find_left_most(int l, const function<bool(info)> &_check)//y=n+1 第二个参数是乱给的
	{
		check = _check;
		z = l - shift; y = n + 1;
		assert(1 <= z && z <= n + 1);
		find_left_most(1, 1, n);
		return {y + shift, res};
	}
	pair<int, info> find_right_most(int r, const function<bool(info)> &_check)//z=0 第二个参数是乱给的
	{
		check = _check;
		z = 0; y = r - shift;
		assert(0 <= y && y <= n);
		find_right_most(1, 1, n);
		return {z + shift, res};
	}
};

struct tag
{
	ll hadd, hc, now;
	bool t;
	void operator+=(const tag &o)
	{
		if (t == 0)
		{
			cmax(hc, o.hc);
			cmax(hadd, now + o.hadd);
		}
		else
			hc = max({hc, now + o.hadd, o.hc});
		if (o.t) now = o.now, t = 1;
		else now += o.now;
	}
};
struct info
{
	ll mx, hmx;
	info operator+(const info &o) const { return {max(mx, o.mx), max(hmx, o.hmx)}; }
	void operator+=(const tag &o)
	{
		hmx = max({hmx, mx + o.hadd, o.hc});
		if (o.t) mx = o.now; else mx += o.now;
	}
};
