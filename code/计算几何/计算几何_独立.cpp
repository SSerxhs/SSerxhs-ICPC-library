#include <bits/stdc++.h>
using namespace std;

// I: ll 坐标，判定/平方量使用 i128；D: long double 坐标。
// I 的空圆判定含四次量，通常建议 |坐标| <= 1e9，避免 i128 溢出。
// 凸包逆时针且不重复首点；直线/半平面有向，半平面保留左侧。
// 别名：geo_ll / geo_db 为二维，geo_3d_ll / geo_3d_db 为三维。
namespace geo
{
	using ll = long long;
	using i128 = __int128_t;
	using db = long double;
	const db eps = 1e-10L, pi = acosl(-1.L), inf = 1e30L;
	mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
	int sgn(ll x) { return (x > 0) - (x < 0); }
	int sgn(i128 x) { return (x > 0) - (x < 0); }
	int sgn(db x) { return (x > eps) - (x < -eps); }
	i128 ab(i128 x) { return x < 0 ? -x : x; }

	namespace I { struct P; }
	namespace D
	{
		struct P
		{
			db x = 0, y = 0;
			P() { }
			P(db x, db y) :x(x), y(y) { }
			explicit P(const I::P &p);
			P operator+() const { return *this; }
			P operator-() const { return {-x, -y}; }
			P operator+(P p) const { return {x + p.x, y + p.y}; }
			P operator-(P p) const { return {x - p.x, y - p.y}; }
			P operator*(db k) const { return {x * k, y * k}; }
			P operator/(db k) const { return {x / k, y / k}; }
			P &operator+=(P p) { x += p.x; y += p.y; return *this; }
			P &operator-=(P p) { x -= p.x; y -= p.y; return *this; }
			P &operator*=(db k) { x *= k; y *= k; return *this; }
			P &operator/=(db k) { x /= k; y /= k; return *this; }
			bool operator==(P p) const { return !sgn(x - p.x) && !sgn(y - p.y); }
			bool operator!=(P p) const { return !(*this == p); }
			bool operator<(P p) const { return x != p.x ? x < p.x : y < p.y; }
			db operator*(P p) const { return x * p.y - y * p.x; }
			db dot(P p) const { return x * p.x + y * p.y; }
			db len2() const { return dot(*this); }
			db len() const { return sqrtl(max<db>(0, len2())); }
			P rot90() const { return {-y, x}; }
			P rot(db a) const { db c = cosl(a), s = sinl(a); return {x * c - y * s, x * s + y * c}; }
			P rotate_90() const { return rot90(); }
			P rotate(db a) const { return rot(a); }
			P unit() const { db l = len(); assert(sgn(l)); return *this / l; }
	};
	P operator*(db k, P p) { return p * k; }
	istream &operator>>(istream &is, P &p) { string x, y; is >> x >> y; p = {stold(x), stold(y)}; return is; }
	ostream &operator<<(ostream &os, P p) { return os << '(' << p.x << ',' << p.y << ')'; }
	db cr(P a, P b) { return a * b; }
	db dt(P a, P b) { return a.dot(b); }
	db cr(P o, P a, P b) { return (a - o) * (b - o); }
	int ori(P o, P a, P b) { return sgn(cr(o, a, b)); }
	db dis2(P a, P b) { return (a - b).len2(); }
	db dis(P a, P b) { return (a - b).len(); }
	db ang(P a, P b) { return atan2l(a * b, a.dot(b)); }
	int qua(P p) { return sgn(p.y) > 0 || !sgn(p.y) && sgn(p.x) >= 0 ? 0 : 1; }
	bool acmp(P a, P b)
	{
		int x = qua(a), y = qua(b);
		return x != y ? x < y : sgn(a * b) ? a * b > 0 : a.len2() < b.len2();
	}

	struct L
	{
		P o, d;
		L() { }
		L(P a, P b) :o(a), d(b - a) { assert(sgn(d.len2())); }
		L(db a, db b, db c) :o(fabsl(a) > fabsl(b) ? P(-c / a, 0) : P(0, -c / b)), d(-b, a) { }
	};
	bool para(L a, L b) { return !sgn(a.d * b.d); }
	bool same(L a, L b) { return para(a, b) && !sgn(a.d * (b.o - a.o)); }
	bool operator==(L a,L b){return same(a,b);}
	bool operator!=(L a,L b){return !same(a,b);}
	bool operator<(L a,L b){int s=sgn(a.d*b.d);return s?s>0:sgn(a.d*a.o-b.d*b.o)<0;}
	ostream&operator<<(ostream&os,L l){return os<<'('<<l.d.x<<"k+"<<l.o.x<<','<<l.d.y<<"k+"<<l.o.y<<')';}
	P inter(L a, L b)
	{
		db z = a.d * b.d; assert(sgn(z));
		return a.o + a.d * ((b.o - a.o) * b.d / z);
	}
	P proj(L l, P p) { return l.o + l.d * ((p - l.o).dot(l.d) / l.d.len2()); }
	P refl(L l, P p) { return proj(l, p) * 2 - p; }
	db dis(L l, P p) { return fabsl(l.d * (p - l.o)) / l.d.len(); }
	db dis(P p, L l) { return dis(l, p); }

	struct S
	{
		P a, b;
		S() { }
		S(P a, P b) :a(a), b(b) { }
		bool cover(P p) const { return !ori(a, b, p) && sgn((p - a).dot(p - b)) <= 0; }
	};
	bool inter(S a, S b)
	{
		int x = ori(a.a, a.b, b.a), y = ori(a.a, a.b, b.b);
		int z = ori(b.a, b.b, a.a), w = ori(b.a, b.b, a.b);
		if (!x && a.cover(b.a) || !y && a.cover(b.b) || !z && b.cover(a.a) || !w && b.cover(a.b)) return 1;
		return x * y < 0 && z * w < 0;
	}
	bool inter(S s, L l) { return sgn(l.d * (s.a - l.o)) * sgn(l.d * (s.b - l.o)) <= 0; }
	bool inter(L l, S s) { return inter(s, l); }
	vector<P> inters(S a, S b)
	{
		if (!inter(a, b)) return { };
		P x = a.b - a.a, y = b.b - b.a;
		if (sgn(x * y)) return {inter(L(a.a, a.b), L(b.a, b.b))};
		vector<P> r;
		for (P p : {a.a, a.b, b.a, b.b}) if (a.cover(p) && b.cover(p)) r.push_back(p);
		sort(r.begin(), r.end()); r.erase(unique(r.begin(), r.end()), r.end());
		return r;
	}
	P near(S s, P p)
	{
		P d = s.b - s.a; db z = d.len2();
		if (!sgn(z)) return s.a;
		db t = (p - s.a).dot(d) / z;
		return s.a + d * clamp(t, 0.L, 1.L);
	}
	db dis(S s, P p) { return dis(near(s, p), p); }
	db dis(P p, S s) { return dis(s, p); }
	db dis(S a, S b)
	{
		if (inter(a, b)) return 0;
		return min({dis(a, b.a), dis(a, b.b), dis(b, a.a), dis(b, a.b)});
	}

	struct C;
	C mec(vector<P> a);
	struct C
	{
		P o; db r = 0;
		C() { }
		C(P o, db r = 0) :o(o), r(r) { }
		C(P a, P b) :o((a + b) / 2), r(dis(a, b) / 2) { }
		C(P a, P b, P c)
		{
			P x = (a + b) / 2, y = (a + c) / 2;
			o = inter(L(x, x + (b - a).rot90()), L(y, y + (c - a).rot90())); r = dis(o, a);
		}
		C(vector<P> a);
		C(const vector<I::P> &a);
		int cover(P p) const { return sgn(r - dis(o, p)); }
	};
	C mec(vector<P> a)
	{
		if (a.empty()) return { };
		shuffle(a.begin(), a.end(), rnd); C c(a[0]);
		for (int i = 1; i < (int)a.size(); ++i) if (c.cover(a[i]) < 0)
		{
			c = C(a[i]);
			for (int j = 0; j < i; ++j) if (c.cover(a[j]) < 0)
			{
				c = C(a[i], a[j]);
				for (int k = 0; k < j; ++k) if (c.cover(a[k]) < 0 && ori(a[i], a[j], a[k])) c = C(a[i], a[j], a[k]);
			}
		}
		return c;
	}
	C::C(vector<P> a) { *this = mec(move(a)); }
	vector<P> inter(C c, L l)
	{
		P p = proj(l, c.o), d = l.d; db z = c.r * c.r - dis2(c.o, p);
		if (sgn(z) < 0) return { };
		if (!sgn(z)) return {p};
		d = d * sqrtl(max<db>(0, z) / d.len2()); return {p - d, p + d};
	}
	vector<P> inter(L l, C c) { return inter(c, l); }
	vector<P> inter(C a, C b)
	{
		P d = b.o - a.o; db l = d.len();
		if (!sgn(l) || sgn(l - a.r - b.r) > 0 || sgn(l - fabsl(a.r - b.r)) < 0) return { };
		db x = (a.r * a.r - b.r * b.r + l * l) / (2 * l), z = a.r * a.r - x * x;
		P p = a.o + d * (x / l);
		if (!sgn(z)) return {p};
		P q = d.rot90() * (sqrtl(max<db>(0, z)) / l); return {p - q, p + q};
	}
	vector<P> inter(C c, S s)
	{
		if (s.a == s.b) return c.cover(s.a) == 0 ? vector<P>{s.a} : vector<P>{};
		vector<P> r; for (P p : inter(c, L(s.a, s.b))) if (s.cover(p)) r.push_back(p); return r;
	}
	vector<P> inter(S s, C c) { return inter(c, s); }
	vector<pair<L, P>> tan(C c, P p)
	{
		vector<pair<L, P>> r; P d = p - c.o; db z = d.len2(), h = z - c.r * c.r;
		if (sgn(h) < 0 || !sgn(z)) return r;
		if (!sgn(h)) { P q = c.o + d * (c.r / sqrtl(z)); return {{L(q, q + d.rot90()), q}}; }
		for (int s : {-1, 1}) { P q = c.o + (d * (c.r * c.r) + d.rot90() * (s * c.r * sqrtl(h))) / z; r.push_back({L(p, q), q}); }
		return r;
	}
	vector<tuple<L, P, P>> tan(C a, C b)
	{
		vector<tuple<L, P, P>> r; P d = b.o - a.o; db z = d.len2();
		if (!sgn(z)) return r;
		for (int o : {-1, 1})
		{
			db x = a.r - o * b.r, h = z - x * x;
			if (sgn(h) < 0) continue;
			h = sqrtl(max<db>(0, h));
			for (int s : {-1, 1})
			{
				P v = (d * x + d.rot90() * (s * h)) / z, p = a.o + v * a.r, q = b.o + v * (o * b.r);
				r.push_back({p == q ? L(p, p + v.rot90()) : L(p, q), p, q});
				if (!sgn(h)) break;
			}
		}
		return r;
	}
	db area(C a, C b)
	{
		db d = dis(a.o, b.o);
		if (sgn(d - a.r - b.r) >= 0) return 0;
		if (sgn(d - fabsl(a.r - b.r)) <= 0) { db r = min(a.r, b.r); return pi * r * r; }
		db x = acosl(clamp((d*d+a.r*a.r-b.r*b.r)/(2*d*a.r),-1.L,1.L));
		db y = acosl(clamp((d*d+b.r*b.r-a.r*a.r)/(2*d*b.r),-1.L,1.L));
		return a.r*a.r*x+b.r*b.r*y-d*a.r*sinl(x);
	}
	db area_tri(C c,P a,P b)
	{
		a-=c.o;b-=c.o;P d=b-a;db A=d.len2(),B=2*a.dot(d),C0=a.len2()-c.r*c.r,z=B*B-4*A*C0;
		vector<db> t{0,1};if(sgn(A)&&sgn(z)>=0){z=sqrtl(max<db>(0,z));for(db x:{(-B-z)/(2*A),(-B+z)/(2*A)})if(sgn(x)>0&&sgn(x-1)<0)t.push_back(x);}
		sort(t.begin(),t.end());db r=0;
		for(int i=1;i<(int)t.size();++i){P x=a+d*t[i-1],y=a+d*t[i],m=(x+y)/2;if(sgn(m.len2()-c.r*c.r)<=0)r+=x*y/2;else r+=c.r*c.r*atan2l(x*y,x.dot(y))/2;}
		return r;
	}
	db area(C c,const struct Poly &a);

	struct Poly
	{
		vector<P> p;
		Poly(vector<P> p = { }) :p(move(p)) { }
		db area2() const
		{
			db r = 0; int n = p.size();
			for (int i = 0; i < n; ++i) r += p[i] * p[(i + 1) % n];
			return r;
		}
		db area() const { return fabsl(area2()) / 2; }
		db peri() const
		{
			db r = 0; int n = p.size();
			for (int i = 0; i < n; ++i) r += dis(p[i], p[(i + 1) % n]);
			return r;
		}
		P cen() const
		{
			db z = 0; P r; int n = p.size();
			for (int i = 0; i < n; ++i) { db c = p[i] * p[(i + 1) % n]; z += c; r += (p[i] + p[(i + 1) % n]) * c; }
			assert(sgn(z)); return r / (3 * z);
		}
		int cover(P q) const
		{
			int r = 0, n = p.size();
			for (int i = 0; i < n; ++i)
			{
				P a = p[i], b = p[(i + 1) % n];
				if (S(a, b).cover(q)) return 0;
				if (sgn(a.y - q.y) > sgn(b.y - q.y)) swap(a, b);
				if (sgn(a.y - q.y) <= 0 && sgn(b.y - q.y) > 0 && ori(a, b, q) > 0) r ^= 1;
			}
			return r ? 1 : -1;
		}
	};
	db area(C c,const Poly &a){db r=0;for(int i=0;i<(int)a.p.size();++i)r+=area_tri(c,a.p[i],a.p[(i+1)%a.p.size()]);return fabsl(r);}
	vector<P> cut(const vector<P> &a, L l)//保留左侧
	{
		vector<P> r; int n = a.size();
		for (int i = 0; i < n; ++i)
		{
			P p = a[i], q = a[(i + 1) % n]; int x = sgn(l.d * (p - l.o)), y = sgn(l.d * (q - l.o));
			if (x >= 0) r.push_back(p);
			if (x * y < 0) r.push_back(inter(L(p, q), l));
		}
		return r;
	}

	vector<P> hull(vector<P> a, bool keep = false)
	{
		sort(a.begin(), a.end()); a.erase(unique(a.begin(), a.end()), a.end());
		if (a.size() <= 1) return a;
		vector<P> r(a.size() * 2); int k = 0;
		for (P p : a)
		{
			while (k > 1 && ori(r[k - 2], r[k - 1], p) < (keep ? 0 : 1)) --k;
			r[k++] = p;
		}
		for (int i = (int)a.size() - 2, t = k; i >= 0; --i)
		{
			while (k > t && ori(r[k - 2], r[k - 1], a[i]) < (keep ? 0 : 1)) --k;
			r[k++] = a[i];
		}
		r.resize(k - 1); return r;
	}
	struct Conv : Poly
	{
		Conv(vector<P> a = { }, bool ready = false) { p = ready ? move(a) : hull(move(a)); }
		int cover(P q) const
		{
			int n = p.size();
			if (!n) return -1;
			if (n == 1) return q == p[0] ? 0 : -1;
			if (n == 2) return S(p[0], p[1]).cover(q) ? 0 : -1;
			int a = ori(p[0], p[1], q), b = ori(p[0], p[n - 1], q);
			if (a < 0 || b > 0) return -1;
			if (!a) return S(p[0], p[1]).cover(q) ? 0 : -1;
			if (!b) return S(p[0], p[n - 1]).cover(q) ? 0 : -1;
			int l = 1, r = n - 1;
			while (l + 1 < r) { int m = (l + r) / 2; if (ori(p[0], p[m], q) >= 0) l = m; else r = m; }
			int s = ori(p[l], p[r], q); return s < 0 ? -1 : !s ? 0 : 1;
		}
	};
	Conv operator+(const Conv &a, const Conv &b)
	{
		if (a.p.empty() || b.p.empty()) return { };
		vector<P> x = a.p, y = b.p;
		auto sh = [](vector<P> &p) { rotate(p.begin(), min_element(p.begin(), p.end()), p.end()); };
		sh(x); sh(y); int n = x.size(), m = y.size();
		vector<P> u(n), v(m), r{x[0] + y[0]};
		for (int i = 0; i < n; ++i) u[i] = x[(i + 1) % n] - x[i];
		for (int i = 0; i < m; ++i) v[i] = y[(i + 1) % m] - y[i];
		for (int i = 0, j = 0; i < n || j < m;)
		{
			if (j == m || i < n && sgn(u[i] * v[j]) > 0) r.push_back(r.back() + u[i++]);
			else if (i == n || sgn(u[i] * v[j]) < 0) r.push_back(r.back() + v[j++]);
			else r.push_back(r.back() + u[i++] + v[j++]);
		}
		r.pop_back(); return Conv(hull(r), true);
	}
	pair<int, int> sector(const Conv &c, P q)
	{
		int n = c.p.size();
		if (!n) return {-1, -1};
		if (n == 1 || q == c.p[0]) return {0, 0};
		if (n == 2) return {0, 1};
		int l = 1, r = n - 1;
		while (l + 1 < r) { int m = (l + r) / 2; if (ori(c.p[0], c.p[m], q) >= 0) l = m; else r = m; }
		if (!ori(c.p[0], c.p[l], q) && sgn((q-c.p[0]).dot(c.p[l]-c.p[0]))>=0) return {l,l};
		if (!ori(c.p[0], c.p[r], q) && sgn((q-c.p[0]).dot(c.p[r]-c.p[0]))>=0) return {r,r};
		return {l, r};
	}
	pair<pair<int, P>, pair<int, P>> tan(const Conv &c, P q)
	{
		int n = c.p.size();
		if (!n || c.cover(q) >= 0) return {{-1,{}},{-1,{}}};
		if (n == 1) return {{0,c.p[0]},{0,c.p[0]}};
		if(n==2)return{{0,c.p[0]},{1,c.p[1]}};
		auto id=[&](long long x){x%=n;if(x<0)x+=n;return(int)x;};
		auto side=[&](int i){return ori(c.p[i],c.p[(i+1)%n],q);};
		auto vis=[&](long long i){return side(id(i))<0;};
		vector<int>v;auto add=[&](int x){x=id(x);if(find(v.begin(),v.end(),x)==v.end())v.push_back(x);};
		auto[x,y]=sector(c,q);for(int i=-4;i<=4;++i)add(x+i),add(y+i);
		int e=-1;for(int x:v)if(vis(x)){e=x;break;}if(e==-1)return{{-1,{}},{-1,{}}};
		long long l,r;
		if(!vis(0)&&!vis(n-1))
		{
			long long a=0,b=e;while(a<b){long long m=(a+b)/2;if(vis(m))b=m;else a=m+1;}l=a;
			a=e;b=n-1;while(a<b){long long m=(a+b+1)/2;if(vis(m))a=m;else b=m-1;}r=a;
		}
		else if(vis(0)&&!vis(n-1))
		{
			l=0;long long a=0,b=n-1;while(a<b){long long m=(a+b+1)/2;if(vis(m))a=m;else b=m-1;}r=a;
		}
		else if(!vis(0)&&vis(n-1))
		{
			r=n-1;long long a=0,b=n-1;while(a<b){long long m=(a+b)/2;if(vis(m))b=m;else a=m+1;}l=a;
		}
		else
		{
			int f=-1;for(int x:v)if(!vis(x)){f=x;break;}
			for(int i=1;f==-1&&i<n;i<<=1)if(!vis(i))f=i;else if(!vis(n-1-i))f=n-1-i;
			if(f==-1&&!vis(n/2))f=n/2;
			if(f==-1)return{{-1,{}},{-1,{}}};
			long long a=0,b=f;while(a<b){long long m=(a+b)/2;if(!vis(m))b=m;else a=m+1;}r=a-1;
			a=f;b=n-1;while(a<b){long long m=(a+b+1)/2;if(!vis(m))a=m;else b=m-1;}l=a+1;
		}
		int a=id(r+1),b=id(l);return{{a,c.p[a]},{b,c.p[b]}};
	}

	struct H
	{
		P o, d;//保留左侧
		H() { }
		H(P a, P b) :o(a), d(b - a) { }
		bool out(P p) const { return sgn(d * (p - o)) < 0; }
	};
	bool hcmp(const H &a, const H &b)
	{
		int x = qua(a.d), y = qua(b.d);
		if (x != y) return x < y;
		int s = sgn(a.d * b.d); return s ? s > 0 : sgn(a.d * (b.o - a.o)) > 0;
	}
	vector<P> hpi(vector<H> a)
	{
		sort(a.begin(), a.end(), hcmp); vector<H> b;
		for (H h : a)
		{
			if (!b.empty() && para(L(b.back().o,b.back().o+b.back().d),L(h.o,h.o+h.d))) b.back() = h;
			else b.push_back(h);
		}
		deque<H> q; deque<P> p;
		for (H h : b)
		{
			while (!p.empty() && h.out(p.back())) p.pop_back(), q.pop_back();
			while (!p.empty() && h.out(p.front())) p.pop_front(), q.pop_front();
			if (!q.empty()) p.push_back(inter(L(q.back().o,q.back().o+q.back().d),L(h.o,h.o+h.d)));
			q.push_back(h);
		}
		while (!p.empty() && q.front().out(p.back())) p.pop_back(), q.pop_back();
		while (!p.empty() && q.back().out(p.front())) p.pop_front(), q.pop_front();
		if (q.size() < 3) return { };
		p.push_back(inter(L(q.back().o,q.back().o+q.back().d),L(q.front().o,q.front().o+q.front().d)));
		return {p.begin(), p.end()};
	}

	pair<int, int> closest(const vector<P> &a)
	{
		int n = a.size(); assert(n >= 2);
		vector<int> id(n); iota(id.begin(), id.end(), 0);
		sort(id.begin(), id.end(), [&](int i, int j) { return a[i] < a[j]; });
		set<pair<db,int>> s; db z = inf; pair<int,int> r{0,1}; int l = 0;
		for (int i : id)
		{
			while (l < n && a[id[l]].x < a[i].x - z) { s.erase({a[id[l]].y,id[l]}); ++l; }
			for (auto it=s.lower_bound({a[i].y-z,-1});it!=s.end()&&it->first<=a[i].y+z;++it)
			{
				db d=dis(a[i],a[it->second]); if(d<z)z=d,r={i,it->second};
			}
			s.insert({a[i].y,i}); if(!sgn(z)) break;
		}
		return r;
	}
	pair<int, int> farthest(const vector<P> &a)
	{
		int n = a.size(); assert(n >= 2); vector<P> p = hull(a); int m = p.size();
		if (m == 1) return {0,1};
		db z = -1; P x,y; int j=1;
		for(int i=0;i<m;++i)
		{
			P d=p[(i+1)%m]-p[i];
			while(fabsl(d*(p[(j+1)%m]-p[i]))>fabsl(d*(p[j]-p[i]))+eps)j=(j+1)%m;
			for(P u:{p[i],p[(i+1)%m]})if(dis2(u,p[j])>z)z=dis2(u,p[j]),x=u,y=p[j];
		}
		int u=find(a.begin(),a.end(),x)-a.begin(),v=find(a.begin(),a.end(),y)-a.begin();
		if(u==v)for(v=0;v<n&&a[v]==x;++v); return {u,v};
	}
	array<P,4> min_rect(const vector<P> &a)
	{
		vector<P> p=hull(a); int n=p.size();
		if(n<3)return {P(),P(),P(),P()};
		db best=inf; array<P,4> ans{}; int j=0,k=0,l=0; P d0=p[1]-p[0];
		for(int x=1;x<n;++x)
		{
			if(dt(d0,p[x]-p[0])>dt(d0,p[j]-p[0]))j=x;
			if(fabsl(cr(d0,p[x]-p[0]))>fabsl(cr(d0,p[k]-p[0])))k=x;
			if(dt(d0,p[x]-p[0])<dt(d0,p[l]-p[0]))l=x;
		}
		for(int i=0;i<n;++i)
		{
			P d=p[(i+1)%n]-p[i];
			while(dt(d,p[(j+1)%n]-p[i])>dt(d,p[j]-p[i])+eps)j=(j+1)%n;
			while(fabsl(cr(d,p[(k+1)%n]-p[i]))>fabsl(cr(d,p[k]-p[i]))+eps)k=(k+1)%n;
			while(dt(d,p[(l+1)%n]-p[i])<dt(d,p[l]-p[i])-eps)l=(l+1)%n;
			db z=(dt(d,p[j]-p[l]))*fabsl(cr(d,p[k]-p[i]))/d.len2();
			if(z<best)
			{
				best=z; P q=d.rot90();
				L a1(p[i],p[i]+d),a2(p[j],p[j]+q),a3(p[k],p[k]+d),a4(p[l],p[l]+q);
				ans={inter(a1,a2),inter(a2,a3),inter(a3,a4),inter(a4,a1)};
			}
		}
		return ans;
	}
	vector<P> lower(vector<P> a)
	{
		sort(a.begin(),a.end()); vector<P> r;
		for(P p:a){while(r.size()>1&&ori(r[r.size()-2],r.back(),p)<=0)r.pop_back();if(r.empty()||sgn(r.back().x-p.x))r.push_back(p);else if(p.y<r.back().y)r.back()=p;}
		return r;
	}
	vector<P> upper(vector<P> a){for(P &p:a)p.y=-p.y;a=lower(a);for(P &p:a)p.y=-p.y;return a;}
	vector<L> lower(vector<L> a)
	{
		sort(a.begin(),a.end(),[](L x,L y){int s=sgn(x.d*y.d);return s?s>0:sgn((x.o-y.o)*y.d)>0;});
		vector<L> r;
		for(L l:a)
		{
			while(!r.empty()&&para(r.back(),l))r.pop_back();
			while(r.size()>1&&sgn((r.back().o-l.o)*r.back().d*(l.d*r[r.size()-2].d)-(r[r.size()-2].o-l.o)*r[r.size()-2].d*(l.d*r.back().d))<=0)r.pop_back();
			r.push_back(l);
		}
		return r;
	}
	vector<L> upper(vector<L> a){for(L &l:a)l.d.y=-l.d.y;a=lower(a);for(L &l:a)l.d.y=-l.d.y;return a;}
	vector<P> pts(const vector<L> &a){vector<P> r;for(int i=1;i<(int)a.size();++i)r.push_back(inter(a[i-1],a[i]));return r;}
	db find_max(const vector<P> &a,db x,db y)
	{
		P q{x,y};int l=0,r=a.size()-1;while(l<r){int m=(l+r)/2;if(a[m].dot(q)<a[m+1].dot(q))l=m+1;else r=m;}return max({a[l].dot(q),a[0].dot(q),a.back().dot(q)});
	}
	db find_min(const vector<P> &a,db x,db y)
	{
		P q{x,y};int l=0,r=a.size()-1;while(l<r){int m=(l+r)/2;if(a[m].dot(q)>a[m+1].dot(q))l=m+1;else r=m;}return min({a[l].dot(q),a[0].dot(q),a.back().dot(q)});
	}
	db max_subset_sum(const vector<P> &a)
	{
		Conv c({P()});for(P p:a)c=c+Conv({P(),p});db r=0;for(P p:c.p)r=max(r,p.len2());return r;
	}

	struct Dyn//下凸壳
	{
		struct cmp{bool operator()(P a,P b)const{return a.x!=b.x?a.x<b.x:a.y<b.y;}};
		set<P,cmp> s;
		void fix(set<P,cmp>::iterator it)
		{
			if(it!=s.begin()){auto j=prev(it);while(j!=s.begin()){auto k=prev(j);if(ori(*k,*j,*it)>0)break;s.erase(j);j=k;}}
			auto j=next(it);while(j!=s.end()){auto k=next(j);if(k==s.end()||ori(*it,*j,*k)>0)break;s.erase(j);j=k;}
		}
		void insert(P p)
		{
			auto it=s.lower_bound({p.x,-inf});
			if(it!=s.end()&&!sgn(it->x-p.x)){if(it->y<=p.y)return;s.erase(it);}
			it=s.lower_bound(p);if(it!=s.begin()&&it!=s.end()&&ori(*prev(it),p,*it)<=0)return;
			it=s.insert(p).first;fix(it);
		}
		int cover(P p)const
		{
			if(s.empty()||p.x<s.begin()->x-eps||p.x>s.rbegin()->x+eps)return -1;
			auto it=s.lower_bound({p.x,-inf});if(it!=s.end()&&!sgn(it->x-p.x))return sgn(p.y-it->y);
			return sgn(cr(*prev(it),*it,p));
		}
	};
	vector<array<int,3>> delaunay(const vector<P> &a)
	{
		int n0=a.size();if(n0<3)return {};
		vector<int> id(n0);iota(id.begin(),id.end(),0);
		sort(id.begin(),id.end(),[&](int i,int j){return a[i]<a[j];});
		id.erase(unique(id.begin(),id.end(),[&](int i,int j){return a[i]==a[j];}),id.end());
		int n=id.size();if(n<3)return {};
		struct E{int r,o,p=-1;};vector<E> e;vector<int> del;e.reserve(n*16);
		auto mk=[&](int u,int v){int z;if(del.empty())z=e.size(),e.resize(z+4);else z=del.back(),del.pop_back();e[z]={z+1,z,u};e[z+1]={z+2,z+3,-1};e[z+2]={z+3,z+2,v};e[z+3]={z,z+1,-1};return z;};
		auto spl=[&](int x,int y){swap(e[e[e[x].o].r].o,e[e[e[y].o].r].o);swap(e[x].o,e[y].o);};
		auto con=[&](int x,int y){int z=mk(e[x^2].p,e[y].p);spl(z,e[e[e[x].r^2].o].r);spl(z^2,y);return z;};
		auto era=[&](int z){spl(z,e[e[e[z].r].o].r);spl(z^2,e[e[e[z^2].r].o].r);z=z>>2<<2;e[z].p=-1;del.push_back(z);};
		auto sol=[&](auto &&sol,int l,int r)->pair<int,int>
		{
			if(r-l==2){int z=mk(id[l],id[l+1]);return {z,z^2};}
			if(r-l==3)
			{
				int x=mk(id[l],id[l+1]),y=mk(id[l+1],id[l+2]);spl(x^2,y);int s=ori(a[id[l]],a[id[l+1]],a[id[l+2]]);
				if(s>0){con(y,x);return {x,y^2};}if(s<0){int z=con(y,x);return {z^2,z};}return {x,y^2};
			}
			int m=(l+r)/2;auto [lo,li]=sol(sol,l,m);auto [ri,ro]=sol(sol,m,r);
			while(1)
			{
				if(ori(a[e[ri].p],a[e[li].p],a[e[li^2].p])>0)li=e[e[e[li].r^2].o].r;
				else if(ori(a[e[li].p],a[e[ri^2].p],a[e[ri].p])>0)ri=e[ri^2].o;else break;
			}
			int bs=con(ri^2,li);if(e[li].p==e[lo].p)lo=bs^2;if(e[ri].p==e[ro].p)ro=bs;
			while(1)
			{
				int u=e[bs^2].p,v=e[bs].p,lc=e[bs^2].o;bool vl=ori(a[u],a[v],a[e[lc^2].p])>0;
				if(vl)while(1){int t=e[lc].o,x=e[t^2].p;if(ori(a[u],a[v],a[x])<=0||C(a[u],a[v],a[e[lc^2].p]).cover(a[x])<=0)break;era(lc);lc=t;}
				int rc=e[e[e[bs].r].o].r;bool vr=ori(a[u],a[v],a[e[rc^2].p])>0;
				if(vr)while(1){int t=e[e[e[rc].r].o].r,x=e[t^2].p;if(ori(a[u],a[v],a[x])<=0||C(a[u],a[v],a[e[rc^2].p]).cover(a[x])<=0)break;era(rc);rc=t;}
				if(!vl&&!vr)break;
				bool go=!vl||vr&&C(a[e[lc^2].p],a[e[lc].p],a[e[rc].p]).cover(a[e[rc^2].p])>0;
				bs=go?con(rc,bs^2):con(bs^2,lc^2);
			}
			return {lo,ro};
		};
		sol(sol,0,n);
		vector<pair<int,int>> eg;
		for(int i=0;i<(int)e.size();i+=4)if(e[i].p!=-1){int u=e[i].p,v=e[i^2].p;if(u<0||v<0||u==v)continue;if(u>v)swap(u,v);eg.push_back({u,v});}
		sort(eg.begin(),eg.end());eg.erase(unique(eg.begin(),eg.end()),eg.end());
		int m=eg.size();vector<vector<pair<int,int>>> g(n0);vector<int> pos(m*2),to(m*2),nxt(m*2),from(m*2);
		for(int i=0;i<m;++i){auto [u,v]=eg[i];from[i*2]=u;to[i*2]=v;from[i*2+1]=v;to[i*2+1]=u;g[u].push_back({v,i*2});g[v].push_back({u,i*2+1});}
		for(int u=0;u<n0;++u)
		{
			sort(g[u].begin(),g[u].end(),[&](auto x,auto y){P p=a[x.first]-a[u],q=a[y.first]-a[u];return acmp(p,q);});
			for(int j=0;j<(int)g[u].size();++j)pos[g[u][j].second]=j;
		}
		for(int i=0;i<m*2;++i){auto &v=g[to[i]];nxt[i]=v[(pos[i^1]+v.size()-1)%v.size()].second;}
		vector<char> vis(m*2);vector<array<int,3>> tr;
		for(int st=0;st<m*2;++st)if(!vis[st])
		{
			vector<int> f;int x=st;
			do{vis[x]=1;f.push_back(from[x]);x=nxt[x];if((int)f.size()>m*2)break;}while(x!=st);
			if(x!=st||f.size()<3)continue;db z=0;for(int i=0;i<(int)f.size();++i)z+=a[f[i]]*a[f[(i+1)%f.size()]];
			if(sgn(z)<=0)continue;for(int i=1;i+1<(int)f.size();++i)if(ori(a[f[0]],a[f[i]],a[f[i+1]])>0)tr.push_back({f[0],f[i],f[i+1]});
		}
		return tr;
	}
	struct DSU
	{
		vector<int> f;DSU(int n=0):f(n){iota(f.begin(),f.end(),0);}int get(int x){return f[x]==x?x:f[x]=get(f[x]);}bool merge(int x,int y){x=get(x);y=get(y);if(x==y)return 0;f[x]=y;return 1;}
	};
	vector<pair<int,int>> emst(const vector<P> &a)
	{
		int n=a.size();if(n<2)return{};auto tr=delaunay(a);vector<tuple<db,int,int>> e;
		vector<int> id(n);iota(id.begin(),id.end(),0);sort(id.begin(),id.end(),[&](int x,int y){return a[x]<a[y];});
		for(int i=1;i<n;++i)if(a[id[i]]==a[id[i-1]])e.push_back({0,id[i-1],id[i]});
		if(tr.empty())for(int i=1;i<n;++i)e.push_back({dis2(a[id[i-1]],a[id[i]]),id[i-1],id[i]});
		for(auto [x,y,z]:tr){e.push_back({dis2(a[x],a[y]),x,y});e.push_back({dis2(a[y],a[z]),y,z});e.push_back({dis2(a[z],a[x]),z,x});}
		sort(e.begin(),e.end());DSU d(n);vector<pair<int,int>> r;for(auto [w,x,y]:e)if(d.merge(x,y))r.push_back({x,y});return r;
	}
	using vec=P;using line=L;using segment=S;using circle=C;using polygon=Poly;using convex=Conv;using half_plane=H;using dynamic_convex=Dyn;
	const P npos{514e194L,9810e191L},apos{145e174L,999e180L},O{0,0};
	int quad(P p){static int d[4]={1,2,4,3};return d[(sgn(p.y)<0)*2+(sgn(p.x)<0)];}
	bool angle_cmp(P a,P b){return acmp(a,b);}
	db get_angle(P a,P b){return ang(a,b);}
	P intersect(L a,L b){if(para(a,b))return same(a,b)?apos:npos;return inter(a,b);}
	bool intersect(S a,S b){return inter(a,b);}
	bool intersect(S s,L l){return inter(s,l);}
	bool intersect(L l,S s){return inter(l,s);}
	vector<P> intersect(C c,L l){return inter(c,l);}
	vector<P> intersect(L l,C c){return inter(l,c);}
	vector<P> intersect(C a,C b){return inter(a,b);}
	vector<P> intersect(C c,S s){return inter(c,s);}
	vector<P> intersect(S s,C c){return inter(s,c);}
	P nearest(S s,P p){return near(s,p);}
	vector<pair<L,P>> tangent(C c,P p){return tan(c,p);}
	vector<tuple<L,P,P>> tangent(C a,C b){return tan(a,b);}
	pair<pair<int,P>,pair<int,P>> tangent(const Conv&c,P p){return tan(c,p);}
	P intersect(H a,H b){return intersect(L(a.o,a.o+a.d),L(b.o,b.o+b.d));}
	vector<P> intersect(vector<H>a)
	{
		db z=1e18L;a.push_back({{-z,-z},{z,-z}});a.push_back({{z,-z},{z,z}});a.push_back({{z,z},{-z,z}});a.push_back({{-z,z},{-z,-z}});return hpi(move(a));
	}
	pair<int,int> closest_pair(const vector<P>&a){return closest(a);}
	pair<int,int> furthest_pair(const vector<P>&a){return farthest(a);}
	array<P,4> rectangle_cover(const vector<P>&a){return min_rect(a);}
	vector<P> convex_down(vector<P>a){return lower(move(a));}
	vector<P> convex_up(vector<P>a){return upper(move(a));}
	vector<L> convex_down(vector<L>a){return lower(move(a));}
	vector<L> convex_up(vector<L>a){return upper(move(a));}
	vector<P> to_vec(const vector<L>&a){return pts(a);}
	vector<pair<int,int>> euclidean_mst(const vector<P>&a){return emst(a);}
	}//namespace D

	namespace I
	{
		struct P
		{
			ll x = 0, y = 0;
			P() { }
			P(ll x, ll y) :x(x), y(y) { }
			explicit P(D::P p) :x(p.x), y(p.y) { }
			P operator+() const { return *this; }
			P operator-() const { return {-x, -y}; }
			P operator+(P p) const { return {x + p.x, y + p.y}; }
			P operator-(P p) const { return {x - p.x, y - p.y}; }
			P operator*(ll k) const { return {x * k, y * k}; }
			P operator/(ll k) const { return {x / k, y / k}; }
			P &operator+=(P p) { x += p.x; y += p.y; return *this; }
			P &operator-=(P p) { x -= p.x; y -= p.y; return *this; }
			P &operator*=(ll k) { x *= k; y *= k; return *this; }
			P &operator/=(ll k) { x /= k; y /= k; return *this; }
			bool operator==(P p) const { return x == p.x && y == p.y; }
			bool operator!=(P p) const { return !(*this == p); }
			bool operator<(P p) const { return x != p.x ? x < p.x : y < p.y; }
			i128 operator*(P p) const { return (i128)x * p.y - (i128)y * p.x; }
			i128 dot(P p) const { return (i128)x * p.x + (i128)y * p.y; }
			i128 len2() const { return dot(*this); }
			db len() const { return sqrtl((db)len2()); }
			P rot90() const { return {-y, x}; }
			P rotate_90() const { return rot90(); }
			D::P rotate(db a) const { db c=cosl(a),s=sinl(a);return{x*c-y*s,x*s+y*c}; }
	};
	P operator*(ll k, P p) { return p * k; }
	istream &operator>>(istream &is, P &p) { return is >> p.x >> p.y; }
	ostream &operator<<(ostream &os, P p) { return os << '(' << p.x << ',' << p.y << ')'; }
	i128 cr(P a, P b) { return a * b; }
	i128 dt(P a, P b) { return a.dot(b); }
	i128 cr(P o, P a, P b) { return (a - o) * (b - o); }
	int ori(P o, P a, P b) { return sgn(cr(o, a, b)); }
	i128 dis2(P a, P b) { return (a - b).len2(); }
	db dis(P a, P b) { return (a - b).len(); }
	db ang(P a, P b) { return atan2l((db)(a * b), (db)a.dot(b)); }
	int qua(P p) { return p.y > 0 || p.y == 0 && p.x >= 0 ? 0 : 1; }
	bool acmp(P a, P b)
	{
		int x = qua(a), y = qua(b); i128 z = a * b;
		return x != y ? x < y : z ? z > 0 : a.len2() < b.len2();
	}

	struct L
	{
		P o, d;
		L() { }
		L(P a, P b) :o(a), d(b - a)
		{
			ll g = gcd(abs(d.x), abs(d.y)); assert(g);
			if (d.x < 0 || !d.x && d.y < 0) g = -g;
			d.x /= g; d.y /= g;
		}
	};
	bool para(L a, L b) { return !(a.d * b.d); }
	bool same(L a, L b) { return para(a, b) && !(a.d * (b.o - a.o)); }
	bool operator==(L a,L b){return same(a,b);}
	bool operator!=(L a,L b){return !same(a,b);}
	bool operator<(L a,L b){i128 z=a.d*b.d;return z?z>0:a.d*a.o<b.d*b.o;}
	ostream&operator<<(ostream&os,L l){return os<<'('<<l.d.x<<"k+"<<l.o.x<<','<<l.d.y<<"k+"<<l.o.y<<')';}
	D::P inter(L a, L b)
	{
		i128 z = a.d * b.d; assert(z);
		db t = (db)((b.o - a.o) * b.d) / (db)z;
		return D::P(a.o.x + a.d.x * t, a.o.y + a.d.y * t);
	}
	struct Rat{P o,d;i128 x,y;};//o+d*x/y
	Rat proj(L l,P p){return {l.o,l.d,(p-l.o).dot(l.d),l.d.len2()};}
	db dis(L l,P p){return fabsl((db)(l.d*(p-l.o)))/l.d.len();}
	db dis(P p,L l){return dis(l,p);}

	struct S
	{
		P a,b;
		S(){ }
		S(P a,P b):a(a),b(b){ }
		bool cover(P p)const{return !ori(a,b,p)&&(p-a).dot(p-b)<=0;}
	};
	bool inter(S a,S b)
	{
		int x=ori(a.a,a.b,b.a),y=ori(a.a,a.b,b.b),z=ori(b.a,b.b,a.a),w=ori(b.a,b.b,a.b);
		if(!x&&a.cover(b.a)||!y&&a.cover(b.b)||!z&&b.cover(a.a)||!w&&b.cover(a.b))return 1;
		return x*y<0&&z*w<0;
	}
	bool inter(S s,L l){return sgn(l.d*(s.a-l.o))*sgn(l.d*(s.b-l.o))<=0;}
	bool inter(L l,S s){return inter(s,l);}
	vector<D::P> inters(S a,S b)
	{
		if(!inter(a,b))return{};if((a.b-a.a)*(b.b-b.a))return{inter(L(a.a,a.b),L(b.a,b.b))};
		vector<P> q;for(P p:{a.a,a.b,b.a,b.b})if(a.cover(p)&&b.cover(p))q.push_back(p);sort(q.begin(),q.end());q.erase(unique(q.begin(),q.end()),q.end());
		vector<D::P> r;for(P p:q)r.push_back(D::P((db)p.x,(db)p.y));return r;
	}
	Rat near(S s,P p)
	{
		P d=s.b-s.a;i128 x=(p-s.a).dot(d),y=d.len2();if(!y||x<=0)return{s.a,d,0,1};if(x>=y)return{s.a,d,1,1};return{s.a,d,x,y};
	}
	db dis(S s,P p)
	{
		if(s.a==s.b)return dis(s.a,p);if((s.b-s.a).dot(p-s.a)<=0)return dis(s.a,p);if((s.a-s.b).dot(p-s.b)<=0)return dis(s.b,p);return dis(L(s.a,s.b),p);
	}
	db dis(P p,S s){return dis(s,p);}
	db dis(S a,S b){if(inter(a,b))return 0;return min({dis(a,b.a),dis(a,b.b),dis(b,a.a),dis(b,a.b)});}

	struct C;
	C mec(vector<P> a);
	struct C
	{
		P a,b,c;short t=1;
		C(P a={}):a(a),t(1){ }
		C(P a,P b):a(a),b(b),t(2){assert(a!=b);}
		C(P a,P b,P c):a(a),b(b),c(c),t(3){assert(ori(a,b,c));if(ori(a,b,c)<0)swap(this->b,this->c);}
		C(vector<P> a);
		int cover(P p)const
		{
			if(t==1)return p==a?0:-1;if(t==2)return -sgn((a-p).dot(b-p));
			P u=a-p,v=b-p,w=c-p;return sgn(u.len2()*(v*w)+v.len2()*(w*u)+w.len2()*(u*v));
		}
	};
	C mec(vector<P> a)
	{
		if(a.empty())return{};shuffle(a.begin(),a.end(),rnd);C c(a[0]);
		for(int i=1;i<(int)a.size();++i)if(c.cover(a[i])<0){c=C(a[i]);for(int j=0;j<i;++j)if(c.cover(a[j])<0){c=C(a[i],a[j]);for(int k=0;k<j;++k)if(c.cover(a[k])<0&&ori(a[i],a[j],a[k]))c=C(a[i],a[j],a[k]);}}
		return c;
	}
	C::C(vector<P> a){*this=mec(move(a));}
	vector<D::P> inter(D::C c,L l){return D::inter(c,D::L(D::P(l.o.x,l.o.y),D::P(l.o.x+l.d.x,l.o.y+l.d.y)));}
	vector<D::P> inter(L l,D::C c){return inter(c,l);}
	vector<D::P> inter(D::C c,S s)
	{
		vector<D::P> r;for(auto p:inter(c,L(s.a,s.b)))if(D::S(D::P(s.a.x,s.a.y),D::P(s.b.x,s.b.y)).cover(p))r.push_back(p);return r;
	}
	vector<D::P> inter(S s,D::C c){return inter(c,s);}
	struct Poly
	{
		vector<P> p;Poly(vector<P> p={}):p(move(p)){ }
		i128 area2()const{i128 r=0;int n=p.size();for(int i=0;i<n;++i)r+=p[i]*p[(i+1)%n];return r;}
		db area()const{return fabsl((db)area2())/2;}
		db peri()const{db r=0;int n=p.size();for(int i=0;i<n;++i)r+=dis(p[i],p[(i+1)%n]);return r;}
		D::P cen()const
		{
			i128 z=0,x=0,y=0;int n=p.size();for(int i=0;i<n;++i){i128 c=p[i]*p[(i+1)%n];z+=c;x+=(i128)(p[i].x+p[(i+1)%n].x)*c;y+=(i128)(p[i].y+p[(i+1)%n].y)*c;}assert(z);return{(db)x/(3*(db)z),(db)y/(3*(db)z)};
		}
		int cover(P q)const
		{
			int r=0,n=p.size();for(int i=0;i<n;++i){P a=p[i],b=p[(i+1)%n];if(S(a,b).cover(q))return 0;if(a.y>b.y)swap(a,b);if(a.y<=q.y&&q.y<b.y&&ori(a,b,q)>0)r^=1;}return r?1:-1;
		}
	};
	vector<D::P> cut(const vector<P> &a,L l)
	{
		vector<D::P>r;int n=a.size();for(int i=0;i<n;++i){P p=a[i],q=a[(i+1)%n];int x=sgn(l.d*(p-l.o)),y=sgn(l.d*(q-l.o));if(x>=0)r.push_back(D::P(p));if(x*y<0)r.push_back(inter(L(p,q),l));}return r;
	}
	vector<P> hull(vector<P> a,bool keep=false)
	{
		sort(a.begin(),a.end());a.erase(unique(a.begin(),a.end()),a.end());if(a.size()<=1)return a;vector<P> r(a.size()*2);int k=0;
		for(P p:a){while(k>1&&ori(r[k-2],r[k-1],p)<(keep?0:1))--k;r[k++]=p;}
		for(int i=a.size()-2,t=k;i>=0;--i){while(k>t&&ori(r[k-2],r[k-1],a[i])<(keep?0:1))--k;r[k++]=a[i];}
		r.resize(k-1);return r;
	}
	struct Conv:Poly
	{
		Conv(vector<P> a={},bool ready=false){p=ready?move(a):hull(move(a));}
		int cover(P q)const
		{
			int n=p.size();if(!n)return-1;if(n==1)return q==p[0]?0:-1;if(n==2)return S(p[0],p[1]).cover(q)?0:-1;
			int a=ori(p[0],p[1],q),b=ori(p[0],p[n-1],q);if(a<0||b>0)return-1;if(!a)return S(p[0],p[1]).cover(q)?0:-1;if(!b)return S(p[0],p[n-1]).cover(q)?0:-1;
			int l=1,r=n-1;while(l+1<r){int m=(l+r)/2;if(ori(p[0],p[m],q)>=0)l=m;else r=m;}int s=ori(p[l],p[r],q);return s<0?-1:!s?0:1;
		}
	};
	Conv operator+(const Conv &a,const Conv &b)
	{
		if(a.p.empty()||b.p.empty())return{};vector<P>x=a.p,y=b.p;auto sh=[](vector<P>&p){rotate(p.begin(),min_element(p.begin(),p.end()),p.end());};sh(x);sh(y);
		int n=x.size(),m=y.size();vector<P>u(n),v(m),r{x[0]+y[0]};for(int i=0;i<n;++i)u[i]=x[(i+1)%n]-x[i];for(int i=0;i<m;++i)v[i]=y[(i+1)%m]-y[i];
		for(int i=0,j=0;i<n||j<m;){i128 z=i<n&&j<m?u[i]*v[j]:0;if(j==m||i<n&&z>0)r.push_back(r.back()+u[i++]);else if(i==n||z<0)r.push_back(r.back()+v[j++]);else r.push_back(r.back()+u[i++]+v[j++]);}r.pop_back();return Conv(hull(r),1);
	}
	pair<int,int> sector(const Conv &c,P q)
	{
		int n=c.p.size();if(!n)return{-1,-1};if(n==1||q==c.p[0])return{0,0};if(n==2)return{0,1};int l=1,r=n-1;while(l+1<r){int m=(l+r)/2;if(ori(c.p[0],c.p[m],q)>=0)l=m;else r=m;}if(!ori(c.p[0],c.p[l],q)&&(q-c.p[0]).dot(c.p[l]-c.p[0])>=0)return{l,l};if(!ori(c.p[0],c.p[r],q)&&(q-c.p[0]).dot(c.p[r]-c.p[0])>=0)return{r,r};return{l,r};
	}
	pair<pair<int,P>,pair<int,P>> tan(const Conv &c,P q)
	{
		int n=c.p.size();if(!n||c.cover(q)>=0)return{{-1,{}},{-1,{}}};if(n==1)return{{0,c.p[0]},{0,c.p[0]}};if(n==2)return{{0,c.p[0]},{1,c.p[1]}};
		auto id=[&](long long x){x%=n;if(x<0)x+=n;return(int)x;};auto side=[&](int i){return ori(c.p[i],c.p[(i+1)%n],q);};auto vis=[&](long long i){return side(id(i))<0;};
		vector<int>v;auto add=[&](int x){x=id(x);if(find(v.begin(),v.end(),x)==v.end())v.push_back(x);};auto[x,y]=sector(c,q);for(int i=-4;i<=4;++i)add(x+i),add(y+i);int e=-1;for(int x:v)if(vis(x)){e=x;break;}if(e==-1)return{{-1,{}},{-1,{}}};long long l,r;
		if(!vis(0)&&!vis(n-1)){long long a=0,b=e;while(a<b){long long m=(a+b)/2;if(vis(m))b=m;else a=m+1;}l=a;a=e;b=n-1;while(a<b){long long m=(a+b+1)/2;if(vis(m))a=m;else b=m-1;}r=a;}
		else if(vis(0)&&!vis(n-1)){l=0;long long a=0,b=n-1;while(a<b){long long m=(a+b+1)/2;if(vis(m))a=m;else b=m-1;}r=a;}
		else if(!vis(0)&&vis(n-1)){r=n-1;long long a=0,b=n-1;while(a<b){long long m=(a+b)/2;if(vis(m))b=m;else a=m+1;}l=a;}
		else{int f=-1;for(int x:v)if(!vis(x)){f=x;break;}for(int i=1;f==-1&&i<n;i<<=1)if(!vis(i))f=i;else if(!vis(n-1-i))f=n-1-i;if(f==-1&&!vis(n/2))f=n/2;if(f==-1)return{{-1,{}},{-1,{}}};long long a=0,b=f;while(a<b){long long m=(a+b)/2;if(!vis(m))b=m;else a=m+1;}r=a-1;a=f;b=n-1;while(a<b){long long m=(a+b+1)/2;if(!vis(m))a=m;else b=m-1;}l=a+1;}
		int a=id(r+1),b=id(l);return{{a,c.p[a]},{b,c.p[b]}};
	}
	struct H
	{
		P o,d;H(){ }H(P a,P b):o(a),d(b-a){ }bool out(P p)const{return d*(p-o)<0;}
	};
	D::P inter(H a,H b){return inter(L(a.o,a.o+a.d),L(b.o,b.o+b.d));}
	vector<D::P> hpi(vector<H> a)
	{
		vector<D::H>b;for(H h:a)b.push_back(D::H(D::P(h.o.x,h.o.y),D::P(h.o.x+h.d.x,h.o.y+h.d.y)));return D::hpi(b);
	}
	pair<int,int> closest(const vector<P>&a)
	{
		int n=a.size();assert(n>=2);vector<int>id(n);iota(id.begin(),id.end(),0);sort(id.begin(),id.end(),[&](int i,int j){return a[i]<a[j];});
		i128 z=dis2(a[0],a[1]);pair<int,int>r{0,1};set<pair<ll,int>>s;int l=0;
		for(int i:id){db d=sqrtl((db)z);while(l<n&&a[id[l]].x<a[i].x-d){s.erase({a[id[l]].y,id[l]});++l;}for(auto it=s.lower_bound({(ll)floorl(a[i].y-d),-1});it!=s.end()&&it->first<=a[i].y+d;++it){i128 q=dis2(a[i],a[it->second]);if(q<z)z=q,r={i,it->second};}s.insert({a[i].y,i});if(!z)break;}return r;
	}
	pair<int,int> farthest(const vector<P>&a)
	{
		int n=a.size();assert(n>=2);vector<P>p=hull(a);int m=p.size();if(m==1)return{0,1};i128 z=-1;P x,y;int j=1;
		for(int i=0;i<m;++i){P d=p[(i+1)%m]-p[i];while(ab(d*(p[(j+1)%m]-p[i]))>ab(d*(p[j]-p[i])))j=(j+1)%m;for(P u:{p[i],p[(i+1)%m]})if(dis2(u,p[j])>z)z=dis2(u,p[j]),x=u,y=p[j];}
		int u=find(a.begin(),a.end(),x)-a.begin(),v=find(a.begin(),a.end(),y)-a.begin();if(u==v)for(v=0;v<n&&a[v]==x;++v);return{u,v};
	}
	array<D::P,4> min_rect(const vector<P>&a)
	{
		vector<P>p=hull(a);int n=p.size();if(n<3)return{};int j=0,k=0,l=0;P d0=p[1]-p[0];
		for(int x=1;x<n;++x){if(dt(d0,p[x]-p[0])>dt(d0,p[j]-p[0]))j=x;if(ab(cr(d0,p[x]-p[0]))>ab(cr(d0,p[k]-p[0])))k=x;if(dt(d0,p[x]-p[0])<dt(d0,p[l]-p[0]))l=x;}
		db best=inf;array<D::P,4>r{};
		for(int i=0;i<n;++i)
		{
			P d=p[(i+1)%n]-p[i];while(dt(d,p[(j+1)%n]-p[i])>dt(d,p[j]-p[i]))j=(j+1)%n;while(ab(cr(d,p[(k+1)%n]-p[i]))>ab(cr(d,p[k]-p[i])))k=(k+1)%n;while(dt(d,p[(l+1)%n]-p[i])<dt(d,p[l]-p[i]))l=(l+1)%n;
			db z=(db)dt(d,p[j]-p[l])*(db)ab(cr(d,p[k]-p[i]))/(db)d.len2();if(z<best){best=z;P q=d.rot90();L a1(p[i],p[i]+d),a2(p[j],p[j]+q),a3(p[k],p[k]+d),a4(p[l],p[l]+q);r={inter(a1,a2),inter(a2,a3),inter(a3,a4),inter(a4,a1)};}
		}
		return r;
	}
	vector<P> lower(vector<P>a)
	{
		sort(a.begin(),a.end());vector<P>r;for(P p:a){while(r.size()>1&&ori(r[r.size()-2],r.back(),p)<=0)r.pop_back();if(r.empty()||r.back().x!=p.x)r.push_back(p);else if(p.y<r.back().y)r.back()=p;}return r;
	}
	vector<P> upper(vector<P>a){for(P&p:a)p.y=-p.y;a=lower(a);for(P&p:a)p.y=-p.y;return a;}
	vector<L> lower(vector<L>a)
	{
		sort(a.begin(),a.end(),[](L x,L y){i128 z=x.d*y.d;return z?z>0:(x.o-y.o)*y.d>0;});vector<L>r;
		for(L l:a){while(!r.empty()&&para(r.back(),l))r.pop_back();while(r.size()>1&&((r.back().o-l.o)*r.back().d)*(l.d*r[r.size()-2].d)-((r[r.size()-2].o-l.o)*r[r.size()-2].d)*(l.d*r.back().d)<=0)r.pop_back();r.push_back(l);}return r;
	}
	vector<L> upper(vector<L>a){for(L&l:a)l.d.y=-l.d.y;a=lower(a);for(L&l:a)l.d.y=-l.d.y;return a;}
	vector<D::P> pts(const vector<L>&a){vector<D::P>r;for(int i=1;i<(int)a.size();++i)r.push_back(inter(a[i-1],a[i]));return r;}
	i128 find_max(const vector<P>&a,ll x,ll y)
	{
		P q{x,y};int l=0,r=a.size()-1;while(l<r){int m=(l+r)/2;if(a[m].dot(q)<a[m+1].dot(q))l=m+1;else r=m;}return max({a[l].dot(q),a[0].dot(q),a.back().dot(q)});
	}
	i128 find_min(const vector<P>&a,ll x,ll y)
	{
		P q{x,y};int l=0,r=a.size()-1;while(l<r){int m=(l+r)/2;if(a[m].dot(q)>a[m+1].dot(q))l=m+1;else r=m;}return min({a[l].dot(q),a[0].dot(q),a.back().dot(q)});
	}
	i128 max_subset_sum(const vector<P>&a){Conv c({P()});for(P p:a)c=c+Conv({P(),p});i128 r=0;for(P p:c.p)r=max(r,p.len2());return r;}
	struct Dyn
	{
		set<P>s;
		void fix(set<P>::iterator it)
		{
			if(it!=s.begin()){auto j=prev(it);while(j!=s.begin()){auto k=prev(j);if(ori(*k,*j,*it)>0)break;s.erase(j);j=k;}}
			auto j=next(it);while(j!=s.end()){auto k=next(j);if(k==s.end()||ori(*it,*j,*k)>0)break;s.erase(j);j=k;}
		}
		void insert(P p)
		{
			auto it=s.lower_bound({p.x,numeric_limits<ll>::min()});if(it!=s.end()&&it->x==p.x){if(it->y<=p.y)return;s.erase(it);}it=s.lower_bound(p);if(it!=s.begin()&&it!=s.end()&&ori(*prev(it),p,*it)<=0)return;it=s.insert(p).first;fix(it);
		}
		int cover(P p)const
		{
			if(s.empty()||p.x<s.begin()->x||p.x>s.rbegin()->x)return-1;auto it=s.lower_bound({p.x,numeric_limits<ll>::min()});if(it!=s.end()&&it->x==p.x)return sgn(p.y-it->y);return sgn(cr(*prev(it),*it,p));
		}
	};
	vector<array<int,3>> delaunay(const vector<P>&a)
	{
		int n0=a.size();if(n0<3)return{};vector<int>id(n0);iota(id.begin(),id.end(),0);sort(id.begin(),id.end(),[&](int i,int j){return a[i]<a[j];});id.erase(unique(id.begin(),id.end(),[&](int i,int j){return a[i]==a[j];}),id.end());int n=id.size();if(n<3)return{};
		struct E{int r,o,p=-1;};vector<E>e;vector<int>del;e.reserve(n*16);
		auto mk=[&](int u,int v){int z;if(del.empty())z=e.size(),e.resize(z+4);else z=del.back(),del.pop_back();e[z]={z+1,z,u};e[z+1]={z+2,z+3,-1};e[z+2]={z+3,z+2,v};e[z+3]={z,z+1,-1};return z;};
		auto spl=[&](int x,int y){swap(e[e[e[x].o].r].o,e[e[e[y].o].r].o);swap(e[x].o,e[y].o);};
		auto con=[&](int x,int y){int z=mk(e[x^2].p,e[y].p);spl(z,e[e[e[x].r^2].o].r);spl(z^2,y);return z;};
		auto era=[&](int z){spl(z,e[e[e[z].r].o].r);spl(z^2,e[e[e[z^2].r].o].r);z=z>>2<<2;e[z].p=-1;del.push_back(z);};
		auto sol=[&](auto&&sol,int l,int r)->pair<int,int>
		{
			if(r-l==2){int z=mk(id[l],id[l+1]);return{z,z^2};}
			if(r-l==3){int x=mk(id[l],id[l+1]),y=mk(id[l+1],id[l+2]);spl(x^2,y);int s=ori(a[id[l]],a[id[l+1]],a[id[l+2]]);if(s>0){con(y,x);return{x,y^2};}if(s<0){int z=con(y,x);return{z^2,z};}return{x,y^2};}
			int m=(l+r)/2;auto[lo,li]=sol(sol,l,m);auto[ri,ro]=sol(sol,m,r);
			while(1){if(ori(a[e[ri].p],a[e[li].p],a[e[li^2].p])>0)li=e[e[e[li].r^2].o].r;else if(ori(a[e[li].p],a[e[ri^2].p],a[e[ri].p])>0)ri=e[ri^2].o;else break;}
			int bs=con(ri^2,li);if(e[li].p==e[lo].p)lo=bs^2;if(e[ri].p==e[ro].p)ro=bs;
			while(1)
			{
				int u=e[bs^2].p,v=e[bs].p,lc=e[bs^2].o;bool vl=ori(a[u],a[v],a[e[lc^2].p])>0;
				if(vl)while(1){int t=e[lc].o,x=e[t^2].p;if(ori(a[u],a[v],a[x])<=0||C(a[u],a[v],a[e[lc^2].p]).cover(a[x])<=0)break;era(lc);lc=t;}
				int rc=e[e[e[bs].r].o].r;bool vr=ori(a[u],a[v],a[e[rc^2].p])>0;
				if(vr)while(1){int t=e[e[e[rc].r].o].r,x=e[t^2].p;if(ori(a[u],a[v],a[x])<=0||C(a[u],a[v],a[e[rc^2].p]).cover(a[x])<=0)break;era(rc);rc=t;}
				if(!vl&&!vr)break;bool go=!vl||vr&&C(a[e[lc^2].p],a[e[lc].p],a[e[rc].p]).cover(a[e[rc^2].p])>0;bs=go?con(rc,bs^2):con(bs^2,lc^2);
			}
			return{lo,ro};
		};
		sol(sol,0,n);vector<pair<int,int>>eg;for(int i=0;i<(int)e.size();i+=4)if(e[i].p!=-1){int u=e[i].p,v=e[i^2].p;if(u<0||v<0||u==v)continue;if(u>v)swap(u,v);eg.push_back({u,v});}sort(eg.begin(),eg.end());eg.erase(unique(eg.begin(),eg.end()),eg.end());
		int m=eg.size();vector<vector<pair<int,int>>>g(n0);vector<int>pos(m*2),to(m*2),nxt(m*2),from(m*2);for(int i=0;i<m;++i){auto[u,v]=eg[i];from[i*2]=u;to[i*2]=v;from[i*2+1]=v;to[i*2+1]=u;g[u].push_back({v,i*2});g[v].push_back({u,i*2+1});}
		for(int u=0;u<n0;++u){sort(g[u].begin(),g[u].end(),[&](auto x,auto y){return acmp(a[x.first]-a[u],a[y.first]-a[u]);});for(int j=0;j<(int)g[u].size();++j)pos[g[u][j].second]=j;}
		for(int i=0;i<m*2;++i){auto&v=g[to[i]];nxt[i]=v[(pos[i^1]+v.size()-1)%v.size()].second;}
		vector<char>vis(m*2);vector<array<int,3>>tr;for(int st=0;st<m*2;++st)if(!vis[st]){vector<int>f;int x=st;do{vis[x]=1;f.push_back(from[x]);x=nxt[x];if((int)f.size()>m*2)break;}while(x!=st);if(x!=st||f.size()<3)continue;i128 z=0;for(int i=0;i<(int)f.size();++i)z+=a[f[i]]*a[f[(i+1)%f.size()]];if(z<=0)continue;for(int i=1;i+1<(int)f.size();++i)if(ori(a[f[0]],a[f[i]],a[f[i+1]])>0)tr.push_back({f[0],f[i],f[i+1]});}return tr;
	}
	struct DSU{vector<int>f;DSU(int n=0):f(n){iota(f.begin(),f.end(),0);}int get(int x){return f[x]==x?x:f[x]=get(f[x]);}bool merge(int x,int y){x=get(x);y=get(y);if(x==y)return 0;f[x]=y;return 1;}};
	vector<pair<int,int>> emst(const vector<P>&a)
	{
		int n=a.size();if(n<2)return{};auto tr=delaunay(a);vector<tuple<i128,int,int>>e;vector<int>id(n);iota(id.begin(),id.end(),0);sort(id.begin(),id.end(),[&](int x,int y){return a[x]<a[y];});for(int i=1;i<n;++i)if(a[id[i]]==a[id[i-1]])e.push_back({0,id[i-1],id[i]});if(tr.empty())for(int i=1;i<n;++i)e.push_back({dis2(a[id[i-1]],a[id[i]]),id[i-1],id[i]});for(auto[x,y,z]:tr){e.push_back({dis2(a[x],a[y]),x,y});e.push_back({dis2(a[y],a[z]),y,z});e.push_back({dis2(a[z],a[x]),z,x});}sort(e.begin(),e.end());DSU d(n);vector<pair<int,int>>r;for(auto[w,x,y]:e)if(d.merge(x,y))r.push_back({x,y});return r;
	}
	using vec=P;using line=L;using segment=S;using circle=C;using polygon=Poly;using convex=Conv;using half_plane=H;using dynamic_convex=Dyn;
	int quad(P p){static int d[4]={1,2,4,3};return d[(p.y<0)*2+(p.x<0)];}
	bool angle_cmp(P a,P b){return acmp(a,b);}
	db get_angle(P a,P b){return ang(a,b);}
	D::P intersect(L a,L b){if(para(a,b))return same(a,b)?D::apos:D::npos;return inter(a,b);}
	bool intersect(S a,S b){return inter(a,b);}
	bool intersect(S s,L l){return inter(s,l);}
	bool intersect(L l,S s){return inter(l,s);}
	vector<D::P> intersect(D::C c,L l){return inter(c,l);}
	vector<D::P> intersect(L l,D::C c){return inter(l,c);}
	vector<D::P> intersect(D::C c,S s){return inter(c,s);}
	vector<D::P> intersect(S s,D::C c){return inter(s,c);}
	Rat nearest(S s,P p){return near(s,p);}
	pair<pair<int,P>,pair<int,P>> tangent(const Conv&c,P p){return tan(c,p);}
	D::P intersect(H a,H b){return inter(a,b);}
	vector<D::P> intersect(vector<H>a){return hpi(move(a));}
	pair<int,int> closest_pair(const vector<P>&a){return closest(a);}
	pair<int,int> furthest_pair(const vector<P>&a){return farthest(a);}
	array<D::P,4> rectangle_cover(const vector<P>&a){return min_rect(a);}
	vector<P> convex_down(vector<P>a){return lower(move(a));}
	vector<P> convex_up(vector<P>a){return upper(move(a));}
	vector<L> convex_down(vector<L>a){return lower(move(a));}
	vector<L> convex_up(vector<L>a){return upper(move(a));}
	vector<D::P> to_vec(const vector<L>&a){return pts(a);}
	vector<pair<int,int>> euclidean_mst(const vector<P>&a){return emst(a);}
	}//namespace I

	D::P::P(const I::P &p):x(p.x),y(p.y){ }
	D::C::C(const vector<I::P> &a)
	{
		vector<D::P> b; b.reserve(a.size()); for (auto p : a) b.push_back(D::P(p)); *this = D::mec(move(b));
	}

	//常用二维补充
	namespace D
	{
		L pbis(P a,P b){P o=(a+b)/2,d=(b-a).rot90();return L(o,o+d);}
		P cc(P a,P b,P c){return C(a,b,c).o;}
		P ic(P a,P b,P c)
		{
			db x=dis(b,c),y=dis(c,a),z=dis(a,b),s=x+y+z;assert(sgn(s));return(a*x+b*y+c*z)/s;
		}
		P ec(P a,P b,P c,int k=0)
		{
			db x=dis(b,c),y=dis(c,a),z=dis(a,b);if(k==0)x=-x;else if(k==1)y=-y;else z=-z;db s=x+y+z;assert(sgn(s));return(a*x+b*y+c*z)/s;
		}
		P hc(P a,P b,P c){return a+b+c-cc(a,b,c)*2;}
		C circum(P a,P b,P c){return C(a,b,c);}
		C inc(P a,P b,P c){P o=ic(a,b,c);return C(o,dis(L(a,b),o));}
		C exc(P a,P b,P c,int k=0){P o=ec(a,b,c,k);return C(o,dis(L(a,b),o));}
		L bis(P o,P a,P b){P d=(a-o).unit()+(b-o).unit();assert(sgn(d.len2()));return L(o,o+d);}
		pair<L,L> bis(L a,L b)
		{
			P o=inter(a,b),x=a.d.unit(),y=b.d.unit();return{L(o,o+x+y),L(o,o+x-y)};
		}
		//0 相离，1 外切，2 相交，3 内切，4 内含，5 重合
		int rel(C a,C b)
		{
			db d=dis(a.o,b.o),s=a.r+b.r,x=fabsl(a.r-b.r);
			if(!sgn(d)&&!sgn(a.r-b.r))return 5;
			if(sgn(d-s)>0)return 0;if(!sgn(d-s))return 1;if(sgn(d-x)>0)return 2;if(!sgn(d-x))return 3;return 4;
		}
		L rad(C a,C b)
		{
			P d=b.o-a.o;assert(sgn(d.len2()));db c=a.o.len2()-b.o.len2()-a.r*a.r+b.r*b.r;return L(2*d.x,2*d.y,c);
		}
		Conv clip(Conv a,const Conv &b)
		{
			if(a.p.empty()||b.p.size()<3)return{};vector<P>r=a.p;for(int i=0;i<(int)b.p.size()&&!r.empty();++i)r=cut(r,L(b.p[i],b.p[(i+1)%b.p.size()]));return Conv(move(r));
		}
		db width(const Conv &c)
		{
			int n=c.p.size();if(n<3)return 0;int j=1;db r=inf;
			for(int i=0;i<n;++i){P d=c.p[(i+1)%n]-c.p[i];while(fabsl(d*(c.p[(j+1)%n]-c.p[i]))>fabsl(d*(c.p[j]-c.p[i]))+eps)j=(j+1)%n;r=min(r,fabsl(d*(c.p[j]-c.p[i]))/d.len());}return r;
		}
		db union_area(const vector<C> &a)
		{
			db ans=0,tau=2*pi;int n=a.size();
			for(int i=0;i<n;++i)
			{
				if(sgn(a[i].r)<=0)continue;bool cov=0;vector<pair<db,db>>v;
				for(int j=0;j<n;++j)if(i!=j&&sgn(a[j].r)>0)
				{
					db d=dis(a[i].o,a[j].o);
					if(sgn(d+a[i].r-a[j].r)<=0&&(sgn(a[j].r-a[i].r)>0||j<i)){cov=1;break;}
					if(!sgn(d)||sgn(d-a[i].r-a[j].r)>=0||sgn(d-fabsl(a[i].r-a[j].r))<=0)continue;
					db o=atan2l(a[j].o.y-a[i].o.y,a[j].o.x-a[i].o.x),z=acosl(clamp((d*d+a[i].r*a[i].r-a[j].r*a[j].r)/(2*d*a[i].r),-1.L,1.L));
					db l=o-z,r=o+z;while(l<0)l+=tau,r+=tau;while(l>=tau)l-=tau,r-=tau;
					if(r<=tau)v.push_back({l,r});else v.push_back({l,tau}),v.push_back({0,r-tau});
				}
				if(cov)continue;sort(v.begin(),v.end());vector<pair<db,db>>u;
				for(auto x:v)if(u.empty()||x.first>u.back().second+eps)u.push_back(x);else u.back().second=max(u.back().second,x.second);
				auto add=[&](db l,db r){db R=a[i].r,x=a[i].o.x,y=a[i].o.y;ans+=(R*x*(sinl(r)-sinl(l))+R*y*(cosl(l)-cosl(r))+R*R*(r-l))/2;};
				db l=0;for(auto x:u){if(l<x.first)add(l,x.first);l=max(l,x.second);}if(l<tau)add(l,tau);
			}
			return fabsl(ans);
		}
	}

	namespace I
	{
		D::P cv(P p){return{(db)p.x,(db)p.y};}
		D::L pbis(P a,P b){D::P o=(cv(a)+cv(b))/2,d=cv(b-a).rot90();return D::L(o,o+d);}
		D::P cc(P a,P b,P c)
		{
			P u=b-a,v=c-a;i128 z=2*(u*v);assert(z);return{a.x+(db)(u.len2()*v.y-v.len2()*u.y)/(db)z,a.y+(db)(v.len2()*u.x-u.len2()*v.x)/(db)z};
		}
		D::P ic(P a,P b,P c){db x=dis(b,c),y=dis(c,a),z=dis(a,b),s=x+y+z;assert(sgn(s));return(cv(a)*x+cv(b)*y+cv(c)*z)/s;}
		D::P ec(P a,P b,P c,int k=0){db x=dis(b,c),y=dis(c,a),z=dis(a,b);if(k==0)x=-x;else if(k==1)y=-y;else z=-z;db s=x+y+z;assert(sgn(s));return(cv(a)*x+cv(b)*y+cv(c)*z)/s;}
		D::P hc(P a,P b,P c){return cv(a)+cv(b)+cv(c)-cc(a,b,c)*2;}
		D::C circum(P a,P b,P c){D::P o=cc(a,b,c);return{o,D::dis(o,cv(a))};}
		D::C inc(P a,P b,P c){D::P o=ic(a,b,c);return{o,D::dis(D::L(cv(a),cv(b)),o)};}
		D::C exc(P a,P b,P c,int k=0){D::P o=ec(a,b,c,k);return{o,D::dis(D::L(cv(a),cv(b)),o)};}
		D::L bis(P o,P a,P b){return D::bis(cv(o),cv(a),cv(b));}
		D::Conv clip(const Conv &a,const Conv &b){vector<D::P>x,y;for(P p:a.p)x.push_back(cv(p));for(P p:b.p)y.push_back(cv(p));return D::clip(D::Conv(x,1),D::Conv(y,1));}
		db width(const Conv &c)
		{
			int n=c.p.size();if(n<3)return 0;int j=1;db r=inf;
			for(int i=0;i<n;++i){P d=c.p[(i+1)%n]-c.p[i];while(ab(d*(c.p[(j+1)%n]-c.p[i]))>ab(d*(c.p[j]-c.p[i])))j=(j+1)%n;r=min(r,(db)ab(d*(c.p[j]-c.p[i]))/d.len());}return r;
		}
		ll lattice_seg(P a,P b){return gcd(abs(a.x-b.x),abs(a.y-b.y))+1;}
		i128 lattice_boundary(const Poly &a){i128 r=0;for(int i=0;i<(int)a.p.size();++i){P x=a.p[i],y=a.p[(i+1)%a.p.size()];r+=gcd(abs(x.x-y.x),abs(x.y-y.y));}return r;}
		i128 lattice_interior(const Poly &a){return(ab(a.area2())-lattice_boundary(a)+2)/2;}
	}

	//简单三维浮点几何
	namespace D3
	{
		struct P
		{
			db x=0,y=0,z=0;P(){ }P(db x,db y,db z):x(x),y(y),z(z){ }
			P operator+()const{return*this;}P operator-()const{return{-x,-y,-z};}
			P operator+(P p)const{return{x+p.x,y+p.y,z+p.z};}P operator-(P p)const{return{x-p.x,y-p.y,z-p.z};}
			P operator*(db k)const{return{x*k,y*k,z*k};}P operator/(db k)const{return{x/k,y/k,z/k};}
			P&operator+=(P p){x+=p.x;y+=p.y;z+=p.z;return*this;}P&operator-=(P p){x-=p.x;y-=p.y;z-=p.z;return*this;}
			P&operator*=(db k){x*=k;y*=k;z*=k;return*this;}P&operator/=(db k){x/=k;y/=k;z/=k;return*this;}
			bool operator==(P p)const{return!sgn(x-p.x)&&!sgn(y-p.y)&&!sgn(z-p.z);}bool operator!=(P p)const{return!(*this==p);}
			db dot(P p)const{return x*p.x+y*p.y+z*p.z;}db len2()const{return dot(*this);}db len()const{return sqrtl(max<db>(0,len2()));}P unit()const{db l=len();assert(sgn(l));return*this/l;}
		};
		P operator*(db k,P p){return p*k;}ostream&operator<<(ostream&os,P p){return os<<'('<<p.x<<','<<p.y<<','<<p.z<<')';}istream&operator>>(istream&is,P&p){string x,y,z;is>>x>>y>>z;p={stold(x),stold(y),stold(z)};return is;}
		db dt(P a,P b){return a.dot(b);}P cr(P a,P b){return{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
		db mix(P a,P b,P c){return a.dot(cr(b,c));}db dis2(P a,P b){return(a-b).len2();}db dis(P a,P b){return(a-b).len();}db ang(P a,P b){return atan2l(cr(a,b).len(),a.dot(b));}
		struct L{P o,d;L(){ }L(P a,P b):o(a),d(b-a){assert(sgn(d.len2()));}};
		struct S{P a,b;S(){ }S(P a,P b):a(a),b(b){ }};
		struct Pl
		{
			P o,n;Pl(){ }Pl(P o,P n):o(o),n(n){assert(sgn(n.len2()));}Pl(P a,P b,P c):o(a),n(cr(b-a,c-a)){assert(sgn(n.len2()));}
			db side(P p)const{return n.dot(p-o);}
		};
		bool para(L a,L b){return!sgn(cr(a.d,b.d).len());}bool para(Pl a,Pl b){return!sgn(cr(a.n,b.n).len());}bool para(L a,Pl b){return!sgn(a.d.dot(b.n));}
		bool perp(L a,L b){return!sgn(a.d.dot(b.d));}bool perp(Pl a,Pl b){return!sgn(a.n.dot(b.n));}bool perp(L a,Pl b){return!sgn(cr(a.d,b.n).len());}
		P proj(L l,P p){return l.o+l.d*((p-l.o).dot(l.d)/l.d.len2());}P refl(L l,P p){return proj(l,p)*2-p;}db dis(L l,P p){return cr(p-l.o,l.d).len()/l.d.len();}db dis(P p,L l){return dis(l,p);}
		P near(S s,P p){P d=s.b-s.a;db z=d.len2();if(!sgn(z))return s.a;return s.a+d*clamp((p-s.a).dot(d)/z,0.L,1.L);}db dis(S s,P p){return dis(near(s,p),p);}db dis(P p,S s){return dis(s,p);}
		db sd(Pl f,P p){return f.side(p)/f.n.len();}db dis(Pl f,P p){return fabsl(sd(f,p));}db dis(P p,Pl f){return dis(f,p);}P proj(Pl f,P p){return p-f.n*(f.side(p)/f.n.len2());}P refl(Pl f,P p){return proj(f,p)*2-p;}
		P inter(L l,Pl f){db z=f.n.dot(l.d);assert(sgn(z));return l.o+l.d*(-f.side(l.o)/z);}P inter(Pl f,L l){return inter(l,f);}
		L inter(Pl a,Pl b)
		{
			P d=cr(a.n,b.n);assert(sgn(d.len2()));db x=a.n.dot(a.o),y=b.n.dot(b.o);P o=cr(b.n*x-a.n*y,d)/d.len2();return L(o,o+d);
		}
		bool inter(S s,Pl f,P &p)
		{
			db x=f.side(s.a),y=f.side(s.b);if(!sgn(x)&&!sgn(y)||sgn(x)*sgn(y)>0)return 0;p=s.a+(s.b-s.a)*(x/(x-y));return 1;
		}
		pair<P,P> near(L a,L b)
		{
			P w=a.o-b.o;db A=a.d.len2(),B=a.d.dot(b.d),C=b.d.len2(),D=a.d.dot(w),E=b.d.dot(w),z=A*C-B*B;
			if(!sgn(z)){P q=proj(b,a.o);return{a.o,q};}db x=(B*E-C*D)/z,y=(A*E-B*D)/z;return{a.o+a.d*x,b.o+b.d*y};
		}
		db dis(L a,L b){auto[p,q]=near(a,b);return dis(p,q);}
		pair<P,P> near(S a,S b)
		{
			P u=a.b-a.a,v=b.b-b.a,w=a.a-b.a;db A=u.len2(),B=u.dot(v),C=v.len2(),D=u.dot(w),E=v.dot(w),x=0,y=0,z=A*C-B*B;
			if(!sgn(A)&&!sgn(C))return{a.a,b.a};if(!sgn(A))y=clamp(E/C,0.L,1.L);else if(!sgn(C))x=clamp(-D/A,0.L,1.L);else{x=sgn(z)?clamp((B*E-C*D)/z,0.L,1.L):0;y=(B*x+E)/C;if(y<0)y=0,x=clamp(-D/A,0.L,1.L);else if(y>1)y=1,x=clamp((B-D)/A,0.L,1.L);}return{a.a+u*x,b.a+v*y};
		}
		db dis(S a,S b){auto[p,q]=near(a,b);return dis(p,q);}
		db area(P a,P b,P c){return cr(b-a,c-a).len()/2;}db vol6(P a,P b,P c,P d){return mix(b-a,c-a,d-a);}db vol(P a,P b,P c,P d){return fabsl(vol6(a,b,c,d))/6;}
		P cen(P a,P b,P c){return(a+b+c)/3;}P cc(P a,P b,P c){P u=b-a,v=c-a,w=cr(u,v);assert(sgn(w.len2()));return a+(cr(v,w)*u.len2()+cr(w,u)*v.len2())/(2*w.len2());}
		P ic(P a,P b,P c){db x=dis(b,c),y=dis(c,a),z=dis(a,b);return(a*x+b*y+c*z)/(x+y+z);}
		bool in_tri(P p,P a,P b,P c)
		{
			P n=cr(b-a,c-a);if(sgn(n.dot(p-a)))return 0;int x=sgn(n.dot(cr(b-a,p-a))),y=sgn(n.dot(cr(c-b,p-b))),z=sgn(n.dot(cr(a-c,p-c)));return x>=0&&y>=0&&z>=0||x<=0&&y<=0&&z<=0;
		}
		P rot(P p,P d,db a){d=d.unit();db c=cosl(a),s=sinl(a);return p*c+cr(d,p)*s+d*(d.dot(p)*(1-c));}P rot(P p,L l,db a){return l.o+rot(p-l.o,l.d,a);}
		struct C{P o;db r=0;C(){ }C(P o,db r):o(o),r(r){ }C(P a,P b):o((a+b)/2),r(dis(a,b)/2){ }C(P a,P b,P c,P d){P u=b-a,v=c-a,w=d-a;db z=2*mix(u,v,w);assert(sgn(z));o=a+(cr(v,w)*u.len2()+cr(w,u)*v.len2()+cr(u,v)*w.len2())/z;r=dis(o,a);}int cover(P p)const{return sgn(r-dis(o,p));}};
		struct Cir{P o,n;db r=0;};
		vector<P> inter(C c,L l){P p=proj(l,c.o),d=l.d;db z=c.r*c.r-dis2(c.o,p);if(sgn(z)<0)return{};if(!sgn(z))return{p};d*=sqrtl(max<db>(0,z)/d.len2());return{p-d,p+d};}
		vector<P> inter(L l,C c){return inter(c,l);}
		bool inter(C c,Pl f,Cir &q){P o=proj(f,c.o);db z=c.r*c.r-dis2(c.o,o);if(sgn(z)<0)return 0;q={o,f.n.unit(),sqrtl(max<db>(0,z))};return 1;}
		bool inter(Pl f,C c,Cir&q){return inter(c,f,q);}
		bool inter(C a,C b,Cir &q)
		{
			P d=b.o-a.o;db l=d.len();if(!sgn(l)||sgn(l-a.r-b.r)>0||sgn(l-fabsl(a.r-b.r))<0)return 0;db x=(a.r*a.r-b.r*b.r+l*l)/(2*l),z=a.r*a.r-x*x;q={a.o+d*(x/l),d/l,sqrtl(max<db>(0,z))};return 1;
		}
		using point=P;using line=L;using segment=S;using plane=Pl;using sphere=C;using circle=Cir;
	}

	//简单三维整数几何：判定保留 i128，非整结果返回 D3 类型。
	namespace I3
	{
		struct P
		{
			ll x=0,y=0,z=0;P(){ }P(ll x,ll y,ll z):x(x),y(y),z(z){ }P operator-()const{return{-x,-y,-z};}P operator+(P p)const{return{x+p.x,y+p.y,z+p.z};}P operator-(P p)const{return{x-p.x,y-p.y,z-p.z};}P operator*(ll k)const{return{x*k,y*k,z*k};}P&operator+=(P p){x+=p.x;y+=p.y;z+=p.z;return*this;}P&operator-=(P p){x-=p.x;y-=p.y;z-=p.z;return*this;}P&operator*=(ll k){x*=k;y*=k;z*=k;return*this;}bool operator==(P p)const{return x==p.x&&y==p.y&&z==p.z;}bool operator!=(P p)const{return!(*this==p);}i128 dot(P p)const{return(i128)x*p.x+(i128)y*p.y+(i128)z*p.z;}i128 len2()const{return dot(*this);}db len()const{return sqrtl((db)len2());}
		};
		P operator*(ll k,P p){return p*k;}ostream&operator<<(ostream&os,P p){return os<<'('<<p.x<<','<<p.y<<','<<p.z<<')';}istream&operator>>(istream&is,P&p){return is>>p.x>>p.y>>p.z;}
		struct Q{i128 x=0,y=0,z=0;Q operator-()const{return{-x,-y,-z};}Q operator+(Q q)const{return{x+q.x,y+q.y,z+q.z};}Q operator-(Q q)const{return{x-q.x,y-q.y,z-q.z};}Q operator*(i128 k)const{return{x*k,y*k,z*k};}};
		i128 dt(P a,P b){return a.dot(b);}Q cr(P a,P b){return{(i128)a.y*b.z-(i128)a.z*b.y,(i128)a.z*b.x-(i128)a.x*b.z,(i128)a.x*b.y-(i128)a.y*b.x};}i128 dt(Q a,P b){return a.x*b.x+a.y*b.y+a.z*b.z;}i128 dt(Q a,Q b){return a.x*b.x+a.y*b.y+a.z*b.z;}db len(Q a){return sqrtl((db)a.x*a.x+(db)a.y*a.y+(db)a.z*a.z);}
		i128 dis2(P a,P b){return(a-b).len2();}db dis(P a,P b){return(a-b).len();}db ang(P a,P b){return atan2l(len(cr(a,b)),(db)a.dot(b));}
		i128 vol6(P a,P b,P c,P d){return dt(cr(b-a,c-a),d-a);}int ori(P a,P b,P c,P d){return sgn(vol6(a,b,c,d));}db area(P a,P b,P c){return len(cr(b-a,c-a))/2;}db vol(P a,P b,P c,P d){return fabsl((db)vol6(a,b,c,d))/6;}
		D3::P cv(P p){return{(db)p.x,(db)p.y,(db)p.z};}
		struct L{P o,d;L(){ }L(P a,P b):o(a),d(b-a){assert(d.len2());}};struct S{P a,b;S(){ }S(P a,P b):a(a),b(b){ }};struct Pl{P o;Q n;Pl(){ }Pl(P o,Q n):o(o),n(n){assert(n.x||n.y||n.z);}Pl(P a,P b,P c):o(a),n(cr(b-a,c-a)){assert(n.x||n.y||n.z);}i128 side(P p)const{return dt(n,p-o);}};
		D3::P proj(L l,P p){db t=(db)(p-l.o).dot(l.d)/(db)l.d.len2();return cv(l.o)+cv(l.d)*t;}db dis(L l,P p){return len(cr(p-l.o,l.d))/l.d.len();}db dis(P p,L l){return dis(l,p);}
		D3::P near(S s,P p){P d=s.b-s.a;i128 x=(p-s.a).dot(d),y=d.len2();if(!y||x<=0)return cv(s.a);if(x>=y)return cv(s.b);return cv(s.a)+cv(d)*((db)x/(db)y);}db dis(S s,P p){return D3::dis(near(s,p),cv(p));}db dis(P p,S s){return dis(s,p);}
		pair<D3::P,D3::P> near(L a,L b){return D3::near(D3::L(cv(a.o),cv(a.o+a.d)),D3::L(cv(b.o),cv(b.o+b.d)));}db dis(L a,L b){auto[p,q]=near(a,b);return D3::dis(p,q);}
		pair<D3::P,D3::P> near(S a,S b){return D3::near(D3::S(cv(a.a),cv(a.b)),D3::S(cv(b.a),cv(b.b)));}db dis(S a,S b){auto[p,q]=near(a,b);return D3::dis(p,q);}
		D3::P inter(L l,Pl f){i128 x=f.side(l.o),y=dt(f.n,l.d);assert(y);db t=-(db)x/(db)y;return cv(l.o)+cv(l.d)*t;}D3::P inter(Pl f,L l){return inter(l,f);}
		bool inter(S s,Pl f,D3::P &p){i128 x=f.side(s.a),y=f.side(s.b);if(!x&&!y||sgn(x)*sgn(y)>0)return 0;p=cv(s.a)+(cv(s.b)-cv(s.a))*((db)x/(db)(x-y));return 1;}
		db dis(Pl f,P p){return fabsl((db)f.side(p))/len(f.n);}D3::P proj(Pl f,P p){db t=(db)f.side(p)/((db)f.n.x*f.n.x+(db)f.n.y*f.n.y+(db)f.n.z*f.n.z);return cv(p)-D3::P((db)f.n.x,(db)f.n.y,(db)f.n.z)*t;}
		D3::P cen(P a,P b,P c){return(cv(a)+cv(b)+cv(c))/3;}D3::P cc(P a,P b,P c){return D3::cc(cv(a),cv(b),cv(c));}D3::P ic(P a,P b,P c){return D3::ic(cv(a),cv(b),cv(c));}
		bool in_tri(P p,P a,P b,P c){Q n=cr(b-a,c-a);if(dt(n,p-a))return 0;i128 x=dt(n,cr(b-a,p-a)),y=dt(n,cr(c-b,p-b)),z=dt(n,cr(a-c,p-c));return x>=0&&y>=0&&z>=0||x<=0&&y<=0&&z<=0;}
		struct C{P o;ll r=0;C(){ }C(P o,ll r):o(o),r(r){ }int cover(P p)const{return sgn((i128)r*r-(p-o).len2());}};
		vector<D3::P> inter(C c,L l){return D3::inter(D3::C(cv(c.o),c.r),D3::L(cv(l.o),cv(l.o+l.d)));}
		using point=P;using line=L;using segment=S;using plane=Pl;using sphere=C;
	}
}//namespace geo

namespace geo_ll=geo::I;
namespace geo_db=geo::D;
namespace geo_3d_ll=geo::I3;
namespace geo_3d_db=geo::D3;
