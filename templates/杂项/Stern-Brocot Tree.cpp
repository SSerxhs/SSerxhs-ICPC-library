template<typename T> struct upcast_t;
template<> struct upcast_t<int> { using type = long long; };
template<> struct upcast_t<long long> { using type = __int128; };
template<typename T> using upcast = typename upcast_t<T>::type;
template<class T> struct frac
{
	T x, y;
	frac<T> operator*(T k) const { return {k * x, k * y}; }
	frac<T> add(const frac<T> &rhs) const { return {x + rhs.x, y + rhs.y}; }
	upcast<T> cross(const frac<T> &o) const { return (upcast<T>)x * o.y - (upcast<T>)y * o.x; }
	T max() const { return ::max(x, y); }
};
template<class T> int cmp(const frac<T> &a, const frac<T> &b)
{
	auto d = a.cross(b);
	return d < 0 ? -1 : !!d;
}
template<class T> struct node
{
	frac<T> l, r;
	frac<T> val() const { return l.add(r); }
	int cmp(const frac<T> &rhs) const { return ::cmp(val(), rhs); }
};
template<class T> node<T> root() { return {{0, 1}, {1, 0}}; }
template<class T> struct sbt
{
	node<T> move(const node<T> &u, char dir, T step) const { return dir == 'L' ? node{u.l, u.r.add(u.l * step)} : node{u.l.add(u.r * step), u.r}; }
	node<T> decode(const vector<pair<char, T>> &a) const
	{
		node u = root<T>();
		for (auto [d, s] : a) u = move(u, d, s);
		return u;
	}
	vector<pair<char, T>> encode(const frac<T> &o) const
	{
		node u = root<T>();
		vector<pair<char, T>> r;
		while (1)
		{
			int d = u.cmp(o);
			if (d == 0) return r;
			char dir = d == 1 ? 'L' : 'R';
			auto kx = u.r.cross(o), ky = o.cross(u.l);
			if (d == 1)
			{
				T k = (kx - 1) / ky;
				r.push_back({'L', k});
				u = move(u, 'L', k);
			}
			else
			{
				T k = (ky - 1) / kx;
				r.push_back({'R', k});
				u = move(u, 'R', k);
			}
		}
	}
	node<T> to_node(const frac<T> &o) const
	{
		node u = root<T>();
		while (1)
		{
			int d = u.cmp(o);
			if (d == 0) return u;
			char dir = d == 1 ? 'L' : 'R';
			auto kx = u.r.cross(o), ky = o.cross(u.l);
			if (d == 1)	u = move(u, 'L', (kx - 1) / ky);
			else u = move(u, 'R', (ky - 1) / kx);
		}
	}
	node<T> lca(const frac<T> &x, const frac<T> &y)
	{
		auto a = encode(x), b = encode(y);
		vector<pair<char, T>> c;
		int sz = min(a.size(), b.size());
		for (int i = 0; i < sz; i++) if (a[i] == b[i]) c.push_back(a[i]);
		else
		{
			if (a[i].first == b[i].first) c.push_back({a[i].first, min(a[i].second, b[i].second)});
			break;
		}
		return decode(c);
	}
	optional<node<T>> jump_to(const frac<T> &x, T depth)//depth(root) = 0
	{
		auto a = encode(x);
		T dep = 0;
		for (auto [x, y] : a) dep += y;
		if (dep < depth) return { };
		T step = dep - depth;
		while (a.size() && a.back().second <= step) step -= a.back().second, a.pop_back();
		if (!step) return {decode(a)};
		a.back().second -= step;
		return {decode(a)};
	}
	optional<node<T>> jump(const frac<T> &x, T step)
	{
		auto a = encode(x);
		while (a.size() && a.back().second <= step) step -= a.back().second, a.pop_back();
		if (!step) return {decode(a)};
		if (!a.size()) return { };
		a.back().second -= step;
		return {decode(a)};
	}
	pair<frac<T>, frac<T>> appro(const frac<T> &x, T n)
	{
		if (x.x <= n && x.y <= n) return {x, x};
		auto a = encode(x);
		auto [l, r] = root<T>();
		for (auto [dir, step] : a) if (dir == 'L')
		{
			frac v = r.add(l * step);
			if (v.add(l).max() <= n) r = v;
			else
			{
				T k = x.x < x.y ? (n - r.y) / l.y : (n - r.x) / l.x;
				return {l, r.add(l * k)};
			}
		}
		else
		{
			frac v = l.add(r * step);
			if (v.add(r).max() <= n) l = v;
			else
			{
				T k = x.x < x.y ? (n - l.y) / r.y : (n - l.x) / r.x;
				return {l.add(r * k), r};
			}
		}
		assert(0);
	}
};
