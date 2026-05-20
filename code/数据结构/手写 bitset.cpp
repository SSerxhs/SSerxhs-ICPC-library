struct Bitset
{
	using ull = unsigned long long;
#define all(x) (x).begin(),(x).end()
	const static ull B = -1llu;
	int n;
	vector<ull> a;
	Bitset() { }
	Bitset(int n) :n(n), a(n + 63 >> 6) { assert(n); }
	bool test(int x) const { assert(x >= 0 && x < n); return a[x >> 6] >> (x & 63) & 1; }
	bool operator[](int x) const { return test(x); }
	void set(int x, bool y) { assert(x >= 0 && x < n); a[x >> 6] = (a[x >> 6] & (B ^ 1llu << (x & 63))) | ((ull)y << (x & 63)); }
	void set(int x) { assert(x >= 0 && x < n); a[x >> 6] |= 1llu << (x & 63); }
	void set() { memset(a.data(), 0xff, a.size() * sizeof a[0]); if (n & 63) a.back() &= (1llu << (n & 63)) - 1; }
	void reset(int x) { assert(x >= 0 && x < n); a[x >> 6] &= ~(1llu << (x & 63)); }
	void reset() { memset(a.data(), 0, a.size() * sizeof a[0]); }
	int count() const
	{
		int r = 0;
		for (ull x : a) r += __builtin_popcountll(x);
		return r;
	}
	int count(int l, int r) const//[l,r)
	{
		if (l == r) return 0;
		if (l >> 6 == r >> 6) return __builtin_popcountll(a[l >> 6] >> (l & 63) & (1llu << r - l) - 1);
		int ans = 0;
		ans += __builtin_popcountll(a[l >> 6] >> (l & 63));
		++(l >>= 6);
		if (r & 63) ans += __builtin_popcountll(a[r >> 6] & (1llu << (r & 63)) - 1);
		r >>= 6;
		while (l < r) ans += __builtin_popcountll(a[l++]);
		return ans;
	}
	Bitset &operator|=(const Bitset &o)
	{
		assert(n == o.n);
		for (int i = 0; i < a.size(); i++) a[i] |= o.a[i];
		return *this;
	}
	Bitset operator|(Bitset o) { o |= *this; return o; }
	Bitset &operator&=(const Bitset &o)
	{
		assert(n == o.n);
		for (int i = 0; i < a.size(); i++) a[i] &= o.a[i];
		return *this;
	}
	Bitset operator&(Bitset o) { o &= *this; return o; }
	Bitset &operator^=(const Bitset &o)
	{
		assert(n == o.n);
		for (int i = 0; i < a.size(); i++) a[i] ^= o.a[i];
		return *this;
	}
	Bitset operator^(Bitset o) { o ^= *this; return o; }
	Bitset operator~() const
	{
		auto r = *this;
		for (ull &x : r.a) x = ~x;
		if (n & 63) r.a.back() &= (1ull << (n & 63)) - 1;
		return r;
	}
	Bitset &operator<<=(int x)
	{
		if (x >= n) return reset(), *this;
		assert(x >= 0);
		int y = x >> 6;
		x &= 63;
		if (x == 0)
		{
			for (int i = (int)a.size() - 1; i >= y; i--) a[i] = a[i - y];
			if (n & 63) a.back() &= (1llu << (n & 63)) - 1;
			memset(a.data(), 0, y * sizeof a[0]);
			return *this;
		}
		for (int i = (int)a.size() - 1; i > y; i--) a[i] = a[i - y] << x | a[i - y - 1] >> 64 - x;
		a[y] = a[0] << x;
		memset(a.data(), 0, y * sizeof a[0]);
		// fill_n(a.begin(),y,0);
		if (n & 63) a.back() &= (1llu << (n & 63)) - 1;
		return *this;
	}
	Bitset operator<<(int x)
	{
		auto r = *this;
		r <<= x;
		return r;
	}
	Bitset &operator>>=(int x)
	{
		if (x >= n) return reset(), *this;
		assert(x >= 0);
		int y = x >> 6, R = (int)a.size() - y - 1;
		x &= 63;
		if (x == 0)
		{
			for (int i = 0; i <= R; i++) a[i] = a[i + y];
			memset(a.data() + R + 1, 0, y * sizeof a[0]);
			return *this;
		}
		for (int i = 0; i < R; i++) a[i] = a[i + y] >> x | a[i + y + 1] << 64 - x;
		a[R] = a.back() >> x;
		memset(a.data() + R + 1, 0, y * sizeof a[0]);
		return *this;
	}
	Bitset operator>>(int x)
	{
		auto r = *this;
		r >>= x;
		return r;
	}
	void range_set(int l, int r)//[l,r) to 1
	{
		if (l == r) return;
		if (l >> 6 == r >> 6)
		{
			a[l >> 6] |= (1llu << r - l) - 1 << (l & 63);
			return;
		}
		if (l & 63)
		{
			a[l >> 6] |= ~((1llu << (l & 63)) - 1);//[l&63,64)
			l += 64;
		}
		if (r & 63) a[r >> 6] |= (1llu << (r & 63)) - 1;
		l >>= 6; r >>= 6;
		memset(a.data() + l, 0xff, (r - l) * sizeof a[0]);
	}
	void range_reset(int l, int r)//[l,r) to 0
	{
		if (l == r) return;
		if (l >> 6 == r >> 6)
		{
			a[l >> 6] &= ~((1llu << r - l) - 1 << (l & 63));
			return;
		}
		if (l & 63)
		{
			a[l >> 6] &= (1llu << (l & 63)) - 1;
			l += 64;
		}
		if (r & 63) a[r >> 6] &= ~((1llu << (r & 63)) - 1);
		l >>= 6; r >>= 6;
		memset(a.data() + l, 0, (r - l) * sizeof a[0]);
	}
	void range_set(int l, int r, bool x)//[l,r)
	{
		if (x) range_set(l, r);
		else range_reset(l, r);
	}
	int size() const { return n; }
	int _Find_first() const
	{
		for (int i = 0; i < a.size(); i++) if (a[i]) return i * 64 + __lg(a[i] & -a[i]);
		return n;
	}
	int _Find_next(int x) const
	{
		assert(x >= 0 && x < n);
		++x;
		if (x == n) return n;
		int y = x & 63; x >>= 6;
		if (a[x] >> y) return x * 64 + __lg(a[x] >> y & -(a[x] >> y)) + y;
		++x;
		while (x < a.size() && !a[x]) ++x;
		return x == a.size() ? n : x * 64 + __lg(a[x] & -a[x]);
	}
	int _Find_last() const
	{
		for (int i = a.size() - 1; i >= 0; i--) if (a[i]) return i * 64 + __lg(a[i]);
		return -1;
	}
	int _Find_prev(int x) const
	{
		assert(x >= 0 && x < n);
		--x;
		if (x == -1) return -1;
		int y = x & 63; x >>= 6;
		if (y < 63)
		{
			if (a[x] & (1llu << y + 1) - 1) return x * 64 + __lg(a[x] & (1llu << y + 1) - 1);
			--x;
		}
		while (x >= 0 && !a[x]) --x;
		return x == -1 ? -1 : x * 64 + __lg(a[x]);
	}
	string to_string() const
	{
		int n = size(), i;
		string s(n, '0');
		for (i = 0; i < n; i++) s[n - i - 1] += test(i);
		return s;
	}
};
istream &operator>>(istream &cin, Bitset &o)
{
	string s;
	cin >> s;
	int n = s.size(), i;
	o.reset();
	assert(n <= o.size());
	for (i = 0; i < n; i++) o.set(i, s[n - i - 1] - '0');
	return cin;
}
ostream &operator<<(ostream &cout, const Bitset &o) { return cout << o.to_string(); }

