struct P
{
	ll tg, L, R;
	P(ll a = 0, ll b = -inf, ll c = inf) :tg(a), L(b), R(c) { }
	void operator+=(P o)
	{
		o.L -= tg; o.R -= tg; tg += o.tg;
		if (L >= o.R) L = R = o.R;
		else if (R <= o.L) L = R = o.L;
		else cmax(L, o.L), cmin(R, o.R);
	}
};
struct Q
{
	ll mx0, cmx, mx1, mn0, cmn, mn1, cnt, sum;
	Q() :mx0(-inf), cmx(0), mx1(-inf), mn0(inf), cmn(0), mn1(inf), cnt(0), sum(0) { }
	Q(ll x) :mx0(x), cmx(1), mx1(-inf), mn0(x), cmn(1), mn1(inf), cnt(1), sum(x) { }
	bool operator+=(const P &o)
	{
		if (o.L == o.R)
		{
			ll c = cnt;
			*this = Q(o.L + o.tg);
			cnt = cmx = cmn = c;
			sum = cnt * (o.L + o.tg);
			return 1;
		}
		if (o.L >= mn1 || o.R <= mx1) return 0;
		if (mx0 == mn0)
		{
			mn0 = min(o.R, max(mx0, o.L));
			sum += cnt * (mn0 - mx0);
			mx0 = mn0;
		}
		else
		{
			if (o.L > mn0)
			{
				sum += (o.L - mn0) * cmn;
				mn0 = o.L;
				cmax(mx1, o.L);
			}
			if (o.R < mx0)
			{
				sum += (o.R - mx0) * cmx;
				mx0 = o.R;
				cmin(mn1, o.R);
			}
		}
		if (o.tg)
		{
			sum += o.tg * cnt;
			mx0 += o.tg;
			mx1 += o.tg;
			mn0 += o.tg;
			mn1 += o.tg;
		}
		return 1;
	}
};
Q operator+(const Q &a, const Q &b)
{
	Q res;
	res.sum = a.sum + b.sum;
	res.cnt = a.cnt + b.cnt;
	res.mx0 = max(a.mx0, b.mx0);
	res.mx1 = max(a.mx1, b.mx1);
	if (res.mx0 == a.mx0) res.cmx += a.cmx; else cmax(res.mx1, a.mx0);
	if (res.mx0 == b.mx0) res.cmx += b.cmx; else cmax(res.mx1, b.mx0);

	res.mn0 = min(a.mn0, b.mn0);
	res.mn1 = min(a.mn1, b.mn1);
	if (res.mn0 == a.mn0) res.cmn += a.cmn; else cmin(res.mn1, a.mn0);
	if (res.mn0 == b.mn0) res.cmn += b.cmn; else cmin(res.mn1, b.mn0);

	return res;
}
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cout << fixed << setprecision(15);
	int n, q, i;
	cin >> n >> q;
	vector<ll> a(n);
	cin >> a;
	sgt<Q, P> s(a.data(), 0, n - 1);
	while (q--)
	{
		int op, l, r;
		cin >> op >> l >> r;
		--r;
		if (op == 3)
		{
			ll res = s.ask(l, r).sum;
			cout << res << '\n';
		}
		else
		{
			ll b;
			cin >> b;
			if (op == 0) s.modify(l, r, {0, -inf, b});
			else if (op == 1) s.modify(l, r, {0, b});
			else s.modify(l, r, {b});
		}
	}
}


	
