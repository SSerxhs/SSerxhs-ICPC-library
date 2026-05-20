
bool FLAG = 0;
ll K;
struct Q
{
	ll x;
	mutable ll y;
	mutable decltype(set<Q>().begin()) r;
	Q operator-(const Q &o) const { return {x - o.x, y - o.y}; }
	lll operator*(const Q &o) const
	{
		return (lll)x * o.y - (lll)y * o.x;
	}
};
set<Q>::iterator END;
bool operator<(const Q &a, const Q &b)
{
	if (FLAG)
	{
		assert(b.x == 0 && b.y == 0);
		if (a.r == END) return 0;
		return (lll)a.x * K + a.y < (lll)a.r->x * K + a.r->y;
	}
	return a.x < b.x;
}
struct convex
{
	set<Q> s;
	ll tagx = 0, tagy = 0;
	void check(auto it)
	{
		decltype(it) jt, kt;
		if (it != s.begin())
		{
			jt = prev(it);
			if (jt != s.begin())
			{
				kt = prev(jt);
				while ((*it - *kt) * (*jt - *kt) <= 0)
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
			while (kt != s.end() && (*kt - *it) * (*jt - *it) <= 0)
				s.erase(jt), jt = kt++;
		}
	}
	void insert(Q p)
	{
		p.y -= tagy + p.x * p.x;
		p.x -= tagx;
		p.y += p.x * p.x;
		auto it = s.lower_bound(p);
		if (it == s.end() || s.begin()->x - p.x > 0) it = s.insert(it, p);
		else if (it->x == p.x) cmax(it->y, p.y);
		else
		{
			auto l = *prev(it);
			if ((p - l) * (*it - l) < 0) it = s.insert(it, p);
		}
		check(it);
		it->r = next(it);
		if (it != s.begin()) prev(it)->r = it;
	}
	void add(ll X, ll Y) { tagx += X; tagy += Y; }
	ll query(ll k)
	{
		k += tagx * 2;
		FLAG = 1; K = k; END = s.end();
		auto it = s.lower_bound({0, 0});
		FLAG = 0;
		return k * (it->x + tagx) + it->y + tagy - tagx * tagx;
	}
	void fun(ll &x, ll &y) const
	{
		y -= x * x;
		x += tagx;
		y += tagy + x * x;
	}
};
