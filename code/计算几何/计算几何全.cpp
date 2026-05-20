namespace geo
{
#define tmpl template<class T>
	using ll = long long;
	using lll = __int128;
	using db = long double;
	tmpl using up = conditional_t<std::is_same_v<T, ll>, lll,
		conditional_t<std::is_same_v<T, db>, db, void>>;
	const db eps = 1e-7, pi = 3.1415926535897932384626434;
#define all(x) (x).begin(),(x).end()
	inline int sgn(const ll &x) { return x < 0 ? -1 : (x > 0); }
	inline int sgn(const lll &x) { return x < 0 ? -1 : (x > 0); }
	inline int sgn(const db &x)
	{
		if (abs(x) < eps) return 0;
		return x > 0 ? 1 : -1;
	}
	tmpl struct vec//* 为叉乘，dot 为点乘，只允许使用 long double 和 ll
	{
		static_assert(std::is_same_v<T, ll> || std::is_same_v<T, db>);
		mutable T x, y;
		vec() { }
		vec(T a, T b) :x(a), y(b) { }
		explicit operator vec<ll>() const { return vec<ll>(x, y); }
		operator vec<db>() const { return vec<db>(x, y); }
		vec<T> operator+(const vec<T> &o) const { return vec(x + o.x, y + o.y); }
		vec<T> operator-(const vec<T> &o) const { return vec(x - o.x, y - o.y); }
		vec<T> operator*(const T &k) const { return vec(x * k, y * k); }
		vec<T> operator/(const T &k) const { return vec(x / k, y / k); }
		T operator*(const vec<T> &o) const { return x * o.y - y * o.x; }
		T dot(const vec<T> &o) const { return x * o.x + y * o.y; }
		void operator+=(const vec<T> &o) { x += o.x; y += o.y; }
		void operator-=(const vec<T> &o) { x -= o.x; y -= o.y; }
		void operator*=(const T &k) { x *= k; y *= k; }
		void operator/=(const T &k) { x /= k; y /= k; }
		bool operator==(const vec<T> &o) const { return x == o.x && y == o.y; }
		bool operator!=(const vec<T> &o) const { return x != o.x || y != o.y; }
		db len() const { return sqrt(len2()); }//模长
		T len2() const { return x * x + y * y; }
		vec<db> rotate(db angle)
		{
			db c = cos(angle), s = sin(angle);
			return vec<db>(x * c - y * s, x * s + y * c);
		}
		vec<T> rotate_90() { return vec<T>(-y, x); }
	};
	const vec<db> npos = vec<db>(514e194, 9810e191), apos = vec<db>(145e174, 999e180), O = vec<db>(0, 0);
	tmpl int quad(const vec<T> &o)//坐标轴归右上象限，返回值 [1,4]
	{
		const static int d[4] = {1, 2, 4, 3};
		return d[(sgn(o.y) < 0) * 2 + (sgn(o.x) < 0)];
	}
	tmpl bool angle_cmp(const vec<T> &a, const vec<T> &b)
	{
		int c = quad(a), d = quad(b);
		if (c != d) return c < d;
		return a * b > 0;
	}
	tmpl db dis(const vec<T> &a, const vec<T> &b) { return (a - b).len(); }
	tmpl T dis2(const vec<T> &a, const vec<T> &b) { return (a - b).len2(); }
	tmpl vec<T> operator*(const T &k, const vec<T> &o) { return vec<T>(k * o.x, k * o.y); }
	tmpl bool operator<(const vec<T> &a, const vec<T> &b)
	{
		int s = sgn(a * b);
		return s > 0 || s == 0 && sgn(a.len2() - b.len2()) < 0;
	}
	istream &operator>>(istream &cin, vec<ll> &o) { return cin >> o.x >> o.y; }
	istream &operator>>(istream &cin, vec<db> &o)
	{
		static string s, t;
		cin >> s >> t;
		o = vec<db>(stod(s), stod(t));
		return cin;
	}
	tmpl ostream &operator<<(ostream &cout, const vec<T> &o)
	{
		if ((vec<db>)o == apos) return cout << "all position";
		if ((vec<db>)o == npos) return cout << "no position";
		return cout << '(' << o.x << ',' << o.y << ')';
	}
	tmpl struct line
	{
		vec<T> o, d;
		line() { }
		line(const vec<T> &a, const vec<T> &b);
		line(db a, db b, db c);
		bool operator!=(const line<T> &m) { return !(*this == m); }
	};
	template<> line<ll>::line(const vec<ll> &a, const vec<ll> &b) :o(a), d(b - a)
	{
		ll tmp = gcd(d.x, d.y);
		assert(tmp);
		if (d.x < 0 || d.x == 0 && d.y < 0) tmp = -tmp;
		d.x /= tmp; d.y /= tmp;
	}
	template<> line<db>::line(const vec<db> &a, const vec<db> &b) :o(a), d(b - a)
	{
		int s = sgn(d.x);
		if (s < 0 || !s && d.y < 0) d.x = -d.x, d.y = -d.y;
		assert(sgn(d.x) || sgn(d.y));
	}
	template<> line<db>::line(db a, db b, db c) : o(abs(a) > abs(b) ? vec<db>(-c / a, 0) : vec<db>(0, -c / b)), d(-b, a) { }//ax+by+c=0
	tmpl db get_angle(const vec<T> &m, const vec<T> &n) { return asin(clamp<db>((m * n) / (m.len() * n.len()), -1, 1)); }
	tmpl bool operator<(const line<T> &m, const line<T> &n)
	{
		int s = sgn(m.d * n.d);
		return s ? s > 0 : m.d * m.o < n.d * n.o;
	}
	tmpl bool operator==(const line<T> &m, const line<T> &n) { return sgn(m.d - n.d) == 0 && sgn((m.o - n.o) * m.d) == 0; }
	tmpl ostream &operator<<(ostream &cout, const line<T> &o) { return cout << '(' << o.d.x << "k + " << o.o.x << " , " << o.d.y << "k + " << o.o.y << ")"; }
	tmpl vec<db> intersect(const line<T> &m, const line<T> &n)
	{
		if (!sgn(m.d * n.d)) return (!sgn(m.d * (n.o - m.o))) ? apos : npos;
		return (vec<db>)m.o + (n.o - m.o) * n.d / (db)(m.d * n.d) * (vec<db>)m.d;
	}
	tmpl db dis(const line<T> &m, const vec<T> &o) { return abs(m.d * (o - m.o) / m.d.len()); }
	tmpl db dis(const vec<T> &o, const line<T> &m) { return abs(m.d * (o - m.o) / m.d.len()); }
	tmpl struct circle;
	template<> struct circle<db>
	{
		vec<db> o;
		db r;
		circle() { }
		circle(const vec<db> &o, const db &r = 0) :o(o), r(r) { }//圆心半径构造
		circle(const vec<db> &a, const vec<db> &b) { o = (a + b) * 0.5; r = dis(b, o); }//直径构造
		circle(const vec<db> &a, const vec<db> &b, const vec<db> &c)//三点构造外接圆（非最小圆）
		{
			auto A = (b + c) * 0.5, B = (a + c) * 0.5;
			o = intersect(line(A, A + (c - A).rotate_90()), line(B, B + (c - B).rotate_90()));
			r = dis(o, c);
		}
		circle(vector<vec<db>> a)
		{
			int n = a.size(), i, j, k;
			mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
			shuffle(all(a), rnd);
			*this = circle(a[0]);
			for (i = 1; i < n; i++) if (cover(a[i]) < 0)
			{
				*this = circle(a[i]);
				for (j = 0; j < i; j++) if (cover(a[j]) < 0)
				{
					*this = circle(a[i], a[j]);
					for (k = 0; k < j; k++) if (cover(a[k]) < 0)
						*this = circle(a[i], a[j], a[k]);
				}
			}
		}
		circle(const vector<vec<ll>> &b)
		{
			vector<vec<db>> a(b.size());
			int n = a.size(), i, j, k;
			for (i = 0; i < a.size(); i++) a[i] = b[i];
			*this = circle(a);
		}
		int cover(const vec<db> &a) const { return sgn(r - dis(a, o)); }//1 表示在内，0 表示在边上，-1 表示在外
	};
	template<> struct circle<ll>
	{
		vec<ll> a, b, c;
		short t;
		circle() : circle({0, 0}) { }
		circle(const vec<ll> &a) :a(a), t(1) { }
		circle(const vec<ll> &a, const vec<ll> &b) :a(a), b(b), t(2) { assert(a != b); }
		circle(const vec<ll> &x, const vec<ll> &y, const vec<ll> &z) :a(x), b(y), c(z), t(3)
		{
			assert(a != b && a != c && b != c);
			if ((b - a) * (c - a) < 0) swap(b, c);
		}
		circle(vector<vec<ll>> a)
		{
			int n = a.size(), i, j, k;
			mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
			shuffle(all(a), rnd);
			*this = circle(a[0]);
			for (i = 1; i < n; i++) if (cover(a[i]) < 0)
			{
				*this = circle(a[i]);
				for (j = 0; j < i; j++) if (cover(a[j]) < 0)
				{
					*this = circle(a[i], a[j]);
					for (k = 0; k < j; k++) if (cover(a[k]) < 0)
						*this = circle(a[i], a[j], a[k]);
				}
			}
		}
		int cover(const vec<ll> &p) const//1 表示在内，0 表示在边上，-1 表示在外
		{
			if (t == 1) return -(a != p);
			if (t == 2) return -sgn((a - p).dot(b - p));
			assert(t == 3);
			vec<ll> u = a - p, v = b - p, w = c - p;
			return sgn((lll)u.len2() * (v * w) + (lll)v.len2() * (w * u) + w.len2() * (u * v));
		}
	};
	tmpl struct segment
	{
		vec<T> a, b;
		segment() { }
		segment(const vec<T> &o, const vec<T> &p) :a(o), b(p)
		{
			int s = sgn(a.x - b.x);
			if (s > 0 || !s && a.y > b.y) swap(a, b);
		}
		bool cover(const vec<T> &o) const { return sgn((o - a) * (b - a)) == 0 && sgn((o - a).dot(o - b)) <= 0; }
	};
	tmpl bool intersect(const segment<T> &m, const segment<T> &n)
	{
		auto a = n.b - n.a, b = m.b - m.a;
		auto d = n.a - m.a;
		if (sgn(n.b.x - m.a.x) < 0 || sgn(m.b.x - n.a.x) < 0) return 0;
		if (sgn(max(n.a.y, n.b.y) - min(m.a.y, m.b.y)) < 0 || sgn(max(m.a.y, m.b.y) - min(n.a.y, n.b.y)) < 0) return 0;
		return sgn(b * d) * sgn((n.b - m.a) * b) >= 0 && sgn(a * d) * sgn((m.b - n.a) * a) <= 0;
	}
	tmpl bool intersect(const segment<T> &m, const line<T> &n) { return sgn(n.d * (m.a - n.o)) * sgn(n.d * (m.b - n.o)) <= 0; }
	tmpl bool intersect(const line<T> &n, const segment<T> &m) { return intersect(m, n); }
	tmpl db dis(const vec<T> &o, const segment<T> &l)
	{
		if (sgn((l.b - l.a).dot(o - l.a)) < 0 || sgn((l.a - l.b).dot(o - l.b)) < 0) return min(dis(o, l.a), dis(o, l.b));
		return dis(o, line(l.a, l.b));
	}
	tmpl db dis(const segment<T> &l, const vec<T> &o) { return dis(o, l); }
	tmpl struct polygon
	{
		vector<vec<T>> p;
		polygon(const vector<vec<T>> &a = { }) :p(a) { }
		db peri() const//周长
		{
			int i, n = p.size();
			if (n == 0) return 0;
			db C = (p[n - 1] - p[0]).len();
			for (i = 1; i < n; i++) C += (p[i - 1] - p[i]).len();
			return C;
		}
		db area() const { return area2() * 0.5; }//面积
		T area2() const//两倍面积
		{
			int i, n = p.size();
			if (n == 0) return 0;
			T S = p[n - 1] * p[0];
			for (i = 1; i < n; i++) S += p[i - 1] * p[i];
			return abs(S);
		}
		int cover(const vec<T> &o) const//点是否在多边形内，-1 外 0 上 1 内
		{
			static mt19937 rnd(75643);
			static uniform_int_distribution<ll> gen(1.2e9, 2e9);
			vec<T> t;
			t.x = gen(rnd); t.y = gen(rnd);
			segment<T> s(o, t);
			int i, n = p.size(), r = 0;
			for (i = 0; i < n; i++)
			{
				if (segment(p[i], p[(i + 1) % n]).cover(o)) return 0;
				r ^= intersect(s, segment(p[i], p[(i + 1) % n]));
			}
			return r ? 1 : -1;
		}
	};
	tmpl struct convex : polygon<T>
	{
		convex(vector<vec<T>> a)
		{
			auto &p = this->p;
			int n = a.size(), i;
			if (!n) return;
			p = a;
			for (i = 1; i < n; i++) if (p[i].x < p[0].x || p[i].x == p[0].x && p[i].y < p[0].y) swap(p[0], p[i]);
			a.resize(0); a.reserve(n);
			for (i = 1; i < n; i++) if (p[i] != p[0]) a.push_back(p[i] - p[0]);
			sort(all(a));
			for (i = 0; i < a.size(); i++) a[i] += p[0];
			vec<T> *st = p.data() - 1;
			int tp = 1;
			for (auto &v : a)
			{
				while (tp > 1 && sgn((st[tp] - st[tp - 1]) * (v - st[tp - 1])) <= 0) --tp;
				st[++tp] = v;
			}
			p.resize(tp);
		}
		int cover(const vec<T> &o) const//点是否在凸包内，-1 外 0 上 1 内
		{
			const auto &p = this->p;
			if (sgn(o.x - p[0].x) < 0 || sgn(o.x - p[0].x) == 0 && sgn(o.y - p[0].y) < 0) return -1;
			if (o == p[0]) return 0;
			if (p.size() == 1) return -1;
			int tmp = sgn((o - p[0]) * (p.back() - p[0]));
			if (tmp == 0) return sgn(dis2(o, p[0]) - dis2(p.back(), p[0])) <= 0 ? 0 : -1;
			if (tmp < 0 || p.size() == 2) return -1;
			int x = upper_bound(1 + all(p), o, [&](const vec<T> &a, const vec<T> &b) { return sgn((a - p[0]) * (b - p[0])) > 0; }) - p.begin() - 1;
			tmp = sgn((o - p[x]) * (p[x + 1] - p[x]));
			if (tmp > 0) return -1;
			return  tmp < 0;
		}
		convex<T> operator+(const convex<T> &A) const
		{
			auto &p = this->p;
			int n = p.size(), m = A.p.size(), i, j;
			vector<vec<T>> a(n), b(m), c;
			for (i = 0; i + 1 < n; i++) a[i] = p[i + 1] - p[i];
			a[n - 1] = p[0] - p[n - 1];
			for (i = 0; i + 1 < m; i++) b[i] = A.p[i + 1] - A.p[i];
			b[m - 1] = A.p[0] - A.p[m - 1];
			c.reserve(n + m);
			c.push_back(p[0] + A.p[0]);
			for (i = j = 0; i < n && j < m;)
			{
				int t = sgn(a[i] * b[j]);
				if (t == 0) c.push_back(c.back() + a[i] + b[j]), ++i, ++j;
				else c.push_back(c.back() + (t > 0 ? a[i++] : b[j++]));
			}
			while (i < n) c.push_back(c.back() + a[i++]);
			while (j < m) c.push_back(c.back() + b[j++]);
			c.pop_back();
			convex<T> t({ });
			t.p = c;
			return t;
		}
	};
	tmpl struct half_plane//默认左侧
	{
		vec<T> o, d;
		operator half_plane<ll>() const { return {(vec<ll>)o, (vec<ll>)(o + d)}; }
		operator half_plane<db>() const { return {(vec<db>)o, (vec<db>)(o + d)}; }
		half_plane() { }
		half_plane(const vec<T> &a, const vec<T> &b) :o(a), d(b - a) { }
		bool operator<(const half_plane<T> &a) const
		{
			int p = quad(d), q = quad(a.d);
			if (p != q) return p < q;
			p = sgn(d * a.d);
			if (p) return p > 0;
			return sgn(d * (a.o - o)) > 0;
		}
	};
	tmpl ostream &operator<<(ostream &cout, half_plane<T> &m) { return cout << m.o << " | " << m.d; }
	tmpl vec<db> intersect(const half_plane<T> &m, const half_plane<T> &n)
	{
		if (!sgn(m.d * n.d)) return sgn(m.d * (n.o - m.o)) ? npos : apos;
		return (vec<db>)m.o + (n.o - m.o) * n.d / (db)(m.d * n.d) * (vec<db>)m.d;
	}
	const db inf = 1e18;
	convex<db> intersect(vector<half_plane<db>> a)
	{
		db I = inf;
		a.push_back({{-I, -I}, {I, -I}});
		a.push_back({{I, -I}, {I, I}});
		a.push_back({{I, I}, {-I, I}});
		a.push_back({{-I, I}, {-I, -I}});
		sort(all(a));
		int n = a.size(), i, h = 0, t = -1;
		vector<half_plane<db>> q(n);
		vector<vec<db>> p(n);
		for (i = 0; i < n; i++) if (i == n - 1 || sgn(a[i].d * a[i + 1].d))
		{
			auto x = (half_plane<db>)a[i];
			while (h < t && sgn((p[t - 1] - x.o) * x.d) >= 0) --t;
			while (h < t && sgn((p[h] - x.o) * x.d) >= 0) ++h;
			q[++t] = x;
			if (h < t) p[t - 1] = intersect(q[t - 1], q[t]);
		}
		while (h < t && sgn((p[t - 1] - q[h].o) * q[h].d) >= 0) --t;
		if (h == t) return convex<db>(vector<vec<db>>(0));
		p[t] = intersect(q[h], q[t]);
		return convex<db>(vector<vec<db>>(p.begin() + h, p.begin() + t + 1));
	}
	tmpl pair<int, int> closest_pair(const vector<vec<T>> &a)
	{
		int n = a.size(), i;
		assert(n >= 2);
		vector<pair<vec<T>, int>> b(n);
		for (i = 0; i < n; i++) b[i] = {a[i], i};
		sort(all(b), [&](auto u, auto v) {
			if (u.first.x != v.first.x) return u.first.x < v.first.x;
			return u.first.y < v.first.y;
		});
		tuple<T, int, int> ans = {dis2(a[0], a[1]), 0, 1};
		set<pair<T, int>> s;
		int j = 0;
		for (auto [v, i] : b)
		{
			auto [x, y] = v;
			T d = sqrtl(get<0>(ans));
			if (d == 0) break;
			for (auto it = s.lower_bound({y - d, 0}); it != s.end() && it->first <= y + d; ++it)
				cmin(ans, tuple{dis2(a[it->second], v), i, it->second});
			s.emplace(v.y, i);
			while (b[j].first.x < v.x - d) s.erase({b[j].first.y, b[j].second}), ++j;
		}
		return {get<1>(ans), get<2>(ans)};
	}
	tmpl pair<int, int> furthest_pair(const vector<vec<T>> &a)
	{
		int n = a.size(), i, j;
		assert(n >= 2);
		auto b = convex(a).p;
		int m = b.size();
		if (m == 1) return {0, 1};
		b.push_back(b[0]);
		tuple<T, int, int> ans{dis2(b[0], b[1]), 0, 1};
		for (i = 0, j = 1; i < m; i++)
		{
			while (abs((b[i + 1] - b[i]) * (b[j] - b[i])) < abs((b[i + 1] - b[i]) * (b[(j + 1) % m] - b[i]))) j = (j + 1) % m;
			cmax(ans, tuple{dis2(b[i], b[j]), i, j});
			cmax(ans, tuple{dis2(b[i + 1], b[j]), i + 1, j});
		}
		auto [_, j1, j2] = ans;
		int i1, i2;
		for (i1 = 0; i1 < n; i1++) if (a[i1] == b[j1]) break;
		for (i2 = 0; i2 < n; i2++) if (i2 != i1 && a[i2] == b[j2]) break;
		return {i1, i2};
	}
	tmpl array<vec<db>, 4> rectangle_cover(const vector<vec<T>> &a)
	{
		const auto &p = convex(a).p;
		int n = p.size(), m = n * 4, i, j, k, l;
		if (n <= 2) return {O, O, O, O};
		vector<vec<T>> b(m);
		for (i = 0; i < m; i++) b[i] = p[i % n];
		tuple<db, int, int, int, int> tmp{inf, 0, 0, 0, 0};
		for (i = j = k = l = 0; i < n * 2; i++)
		{
			cmax(j, i + 1);
			auto d = b[i + 1] - b[i];
			while (d.dot(b[j] - b[i]) < d.dot(b[j + 1] - b[i])) ++j;
			while (j > i && d.dot(b[j] - b[i]) < d.dot(b[j - 1] - b[i])) --j;
			cmax(k, j);
			while (abs(d * (b[k] - b[i])) < abs(d * (b[k + 1] - b[i]))) ++k;
			while (k > j && abs(d * (b[k] - b[i])) < abs(d * (b[k - 1] - b[i]))) --k;
			cmax(l, k);
			while (d.dot(b[l] - b[i]) > d.dot(b[l + 1] - b[i])) ++l;
			while (l > k && d.dot(b[l] - b[i]) > d.dot(b[l - 1] - b[i])) --l;
			assert(l + 1 < m);
			if (i >= n) cmin(tmp, tuple{(db)(b[j] - b[l]).dot(d) * abs((b[k] - b[i]) * d) / d.len2(), i, j, k, l});
		}
		tie(ignore, i, j, k, l) = tmp;
		auto d = b[i + 1] - b[i], rd = d.rotate_90();
		line l1(b[i], b[i] + d), l2(b[j], b[j] + rd), l3(b[k], b[k] + d), l4(b[l], b[l] + rd);
		return {intersect(l1, l2), intersect(l2, l3), intersect(l3, l4), intersect(l4, l1)};
	}
	tmpl vector<line<T>> convex_up(vector<line<T>> a)
	{
		for (auto &t : a) t.d.y = -t.d.y;
		a = convex_down(a);
		for (auto &t : a) t.d.y = -t.d.y;
		return a;
	}
	tmpl vector<line<T>> convex_down(vector<line<T>> a)
	{
		sort(all(a), [&](const auto &u, const auto &v) {
			int t = sgn(u.d * v.d);
			return t ? t > 0 : sgn((u.o - v.o) * v.d) > 0;
		});
		vector<line<T>> b;
		int tp = -1;
		for (auto t : a)
		{
			while (tp >= 0 && sgn(b[tp].d * t.d) == 0) --tp, b.pop_back();
			while (tp >= 1 && sgn((up<T>)((b[tp].o - t.o) * b[tp].d) * (t.d * b[tp - 1].d) - (up<T>)((b[tp - 1].o - t.o) * b[tp - 1].d) * (t.d * b[tp].d)) <= 0)
				--tp, b.pop_back();
			++tp; b.push_back(t);
		}
		return b;
	}
	tmpl vector<vec<T>> convex_down(vector<vec<T>> a)
	{
		sort(all(a), [&](const auto &u, const auto &v) {
			int t = sgn(u.x - v.x);
			if (t) return t < 0;
			return u.y > v.y;
		});
		vector<vec<T>> b;
		int tp = -1;
		for (auto t : a)
		{
			while (tp >= 0 && sgn(b[tp].x - t.x) == 0) --tp, b.pop_back();
			while (tp >= 1 && sgn((t - b[tp]) * (t - b[tp - 1])) >= 0)
				--tp, b.pop_back();
			++tp; b.push_back(t);
		}
		return b;
	}
	tmpl vector<vec<T>> convex_up(vector<vec<T>> a)
	{
		for (auto &t : a) t.d.y = -t.d.y;
		a = convex_down(a);
		for (auto &t : a) t.d.y = -t.d.y;
		return a;
	}
	tmpl vector<vec<db>> to_vec(const vector<line<T>> &a)
	{
		int n = a.size(), i;
		vector<vec<db>> b(n - 1);
		for (i = 0; i < n - 1; i++) b[i] = intersect(a[i], a[i + 1]);
		return b;
	}
	tmpl T find_max(const vector<vec<T>> &a, T kx, T ky)//要求函数凸
	{
		vec<T> p = {kx, ky};
		int l = 0, r = a.size() - 1, mid;
		while (l < r)
		{
			mid = (l + r) / 2;
			if (a[mid].dot(p) < a[mid + 1].dot(p)) l = mid + 1;
			else r = mid;
		}
		return max({a[l].dot(p), a[0].dot(p), a.back().dot(p)});
	}
	tmpl T find_min(const vector<vec<T>> &a, T kx, T ky)//要求函数凸
	{
		vec<T> p = {kx, ky};
		int l = 0, r = a.size() - 1, mid;
		while (l < r)
		{
			mid = (l + r) / 2;
			if (a[mid].dot(p) > a[mid + 1].dot(p)) l = mid + 1;
			else r = mid;
		}
		return min({a[l].dot(p), a[0].dot(p), a.back().dot(p)});
	}
	tmpl T max_subset_sum(const vector<vec<T>> &a)
	{
		int n = a.size(), i;
		function<convex<T>(int, int)> dfs = [&](int l, int r) {
			if (l + 1 == r) return convex<T>({vec<T>(0, 0), a[l]});
			int mid = (l + r) / 2;
			return dfs(l, mid) + dfs(mid, r);
		};
		const auto &p = dfs(0, n).p;
		T ans = 0;
		for (auto t : p) ans = max(ans, t.len2());
		return ans;
	}
	template<class T> struct dynamic_convex//下凸
	{
		set < vec<T>, decltype([](const vec<T> &a, const vec<T> &b) {
			return sgn(a.x - b.x) < 0;
		}) > s;
		void check(auto it)
		{
			decltype(it) jt, kt;
			if (it != s.begin())
			{
				jt = prev(it);
				if (jt != s.begin())
				{
					kt = prev(jt);
					while (sgn((*it - *kt) * (*jt - *kt)) >= 0)
					{
						s.erase(jt);
						if (kt == s.begin()) break;
						jt = kt--;
					}
				}
			}
			jt = next(it);
			if (jt != s.end())
			{
				kt = next(jt);
				while (kt != s.end() && sgn((*kt - *it) * (*jt - *it)) >= 0)
					s.erase(jt), jt = kt++;
			}
		}
		void insert(const vec<T> &p)
		{
			auto it = s.lower_bound(p);
			if (it == s.end() || sgn(s.begin()->x - p.x) > 0) it = s.insert(it, p);
			else if (sgn(it->x - p.x) == 0) cmin(it->y, p.y);
			else
			{
				auto l = *prev(it);
				if (sgn((p - l) * (*it - l)) > 0) it = s.insert(it, p);
			}
			check(it);
		}
		int cover(const vec<T> &p)//在凸壳区域以上返回 1，在凸壳上返回 0，其余返回 -1。
		{
			if (s.size() == 0) return -1;
			if (sgn(p.x - s.begin()->x) < 0 || sgn(p.x - s.rbegin()->x) > 0) return -1;
			auto it = s.lower_bound(p);
			if (sgn(it->x - p.x) == 0) return sgn(p.y - it->y);
			auto l = *prev(it);
			return sgn((*it - l) * (p - l));
		}
	};
#undef tmpl
}
using geo::vec, geo::line, geo::circle, geo::convex, geo::polygon, geo::half_plane;
using geo::eps, geo::pi, geo::segment, geo::sgn, geo::dynamic_convex;
using Q = vec<ll>;
void read(db &x) { static string s; cin >> s; x = stod(s); }
template<typename T, typename... Args> void read(T &first, Args&... args) { read(first); read(args...); }
