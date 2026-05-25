const db eps = 1e-7;
db sl, sr, sm, a;
db f(db x)
{
	return pow(x, a / x - x);
}
db g(db l, db r)
{
	db mid = (l + r) * 0.5;
	return (f(l) + f(r) + f(mid) * 4) / 6 * (r - l);
}
db sim(db l, db r)
{
	db mid = (l + r) * 0.5;
	sl = g(l, mid); sr = g(mid, r); sm = g(l, r);
	if (abs(sl + sr - sm) < eps) return sl + sr;
	return sim(l, mid) + sim(mid, r);
}
