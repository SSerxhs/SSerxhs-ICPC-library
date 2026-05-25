template<class T, int M = sizeof(T) * 8> struct base//线性基
{
	array<T, M> a;
	int num, dim;
	base() :a{ }, num(0), dim(0) { }
	bool insert(T x)//线性基插入
	{
		++num;
		if (x == 0) return 0;
		for (int i = __lg(x); x; i = __lg(x))
		{
			if (!a[i]) return ++dim, a[i] = x;
			x ^= a[i];
		}
		return 0;
	}
	bool contains(T x, bool empty_allowed = true) const//查询是否能 xor 出 x
	{
		if (x == 0) return empty_allowed || dim != num;
		for (int i = __lg(x); x; i = __lg(x))
		{
			if (!a[i]) return 0;
			x ^= a[i];
		}
		return 1;
	}
	T max(T x = 0, T k = 0, bool empty_allowed = true) const//查询子集 xor 的 k+1 大。若有传入参数 x，表示子集 xor x 的 k+1 大。 
	{
		assert(M <= 64);
		bool zero = empty_allowed || num != dim;
		if (dim < 64 && k >= (1ull << dim) - !zero || k == (-1ull) && !zero)
		{
			assert(M != sizeof(T) * 8);
			return -1;
		}
		if (!zero)
		{
			ull z = 0;
			int d = dim;
			for (int i = M - 1; i >= 0; i--) if (a[i]) z |= (1ull ^ (x >> i & 1)) << --d;
			if (k >= z) ++k;
		}
		int d = dim;
		for (int i = M - 1; i >= 0; i--)
			if (a[i] && (1 ^ (k >> --d ^ x >> i) & 1))
				x ^= a[i];
		return x;
	}
	T min(T x = 0, T k = 0, bool empty_allowed = true) const//查询子集 xor 的 k+1 大。若有传入参数 x，表示子集 xor x 的 k+1 大。 
	{
		assert(M <= 64);
		bool zero = empty_allowed || num != dim;
		if (dim == 0 && k >= zero || dim < 64 && k >= (1ull << dim) - !zero || k == (-1ull) && !zero)
		{
			assert(M != sizeof(T) * 8);
			return -1;
		}
		if (!zero)
		{
			ull z = 0;
			int d = dim;
			for (int i = M - 1; i >= 0; i--) if (a[i]) z |= (0ull + (x >> i & 1)) << --d;
			if (k >= z) ++k;
		}
		int d = dim;
		for (int i = M - 1; i >= 0; i--)
			if (a[i] && ((k >> --d ^ x >> i) & 1))
				x ^= a[i];
		return x;
	}
	base &operator|=(const base &o)//合并线性基
	{
		int t = num;
		for (T x : o.a) if (x) insert(x);
		num = t + o.num;
		return *this;
	}
	base operator|(base o) const { return o += *this; }//合并线性基
	base operator&(base o) const
	{
		array<T, M> g1{ }, g2{ }, val{ };
		int i, j, k;
		for (i = 0; i < M; i++)
			for (j = 0; j < M; j++)
				g1[i] |= (a[j] >> i & 1) << j,
				g2[i] |= (o.a[j] >> i & 1) << j;

		T mask1 = 0, mask2 = 0;
		for (i = M - 1; i >= 0; i--) if (g1[i] || g2[i])
		{
			auto &x = (g1[i] ? g1 : g2);
			int g = __lg(x[i]);
			(g1[i] ? mask1 : mask2) |= (T)1 << g;
			if (g1[i]) val[i] = a[g];
			for (j = 0; j < M; j++)
				if (i != j && (x[j] >> g & 1))
					g1[j] ^= g1[i], g2[j] ^= g2[i];
		}
		base<T, M> res;
		for (i = 0; i < M; i++) if (a[i] && !(mask1 >> i & 1))
		{
			T v = a[i];
			for (j = 0; j < M; j++)
				if (g1[j] >> i & 1)
					v ^= val[j];
			res.insert(v);
		}
		for (i = 0; i < M; i++) if (o.a[i] && !(mask2 >> i & 1))
		{
			T v = 0;
			for (j = 0; j < M; j++)
				if (g2[j] >> i & 1)
					v ^= val[j];
			res.insert(v);
		}
		return res;
	}
	base &operator&=(const base &o) { return *this = *this & o; }

};
template<class T = ll, int M = sizeof(T) * 8> struct rangebase//[0,...)
{
	vector<array<pair<T, int>, M>> a;
	rangebase() :a{{ }} { }
	rangebase(const vector<T> &b) :a{{ }} { for (T x : b) push_back(x); }//直接用一个 vector 构造
	void push_back(T x)//在最后插入 x
	{
		int n = a.size() - 1;
		a.push_back(a.back());
		if (x == 0) return;
		for (int i = __lg(x); x; i = __lg(x))
		{
			auto &[v, p] = a.back()[i];
			if (v)
			{
				if (n > p)
				{
					swap(x, v);
					swap(n, p);
				}
				x ^= v;
			}
			else
			{
				v = x;
				p = n;
				return;
			}
		}
	}
	base<T, M> ask(int l, int r)//查询 $[l,r)$ 元素构成的线性基。下标从 0 开始（同 vector）
	{
		assert(0 <= l && l <= r && r <= a.size());
		base<T, M> res;
		res.num = r - l;
		for (int i = 0; i < M; i++)
		{
			auto [v, p] = a[r][i];
			if (v && p >= l) res.a[i] = v, ++res.dim;
		}
		return res;
	}
};

