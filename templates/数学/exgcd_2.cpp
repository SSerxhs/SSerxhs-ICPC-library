pair<ll, ll> exgcd(ll a, ll b, ll c)//ax+by=c，{-1,-1} 无解，b=0 返回 {c/a,0}，否则返回最小非负 x
{
	assert(a || b);
	ll d = gcd(a, b);
	if (c % d) return {-1, -1};
	if (!b) return {c / a, 0};
	if (a < 0) a = -a, b = -b, c = -c;
	ll x = 1, x1 = 0, p = a, q = b, k;
	b = abs(b);
	while (b)
	{
		k = a / b;
		x -= k * x1; a -= k * b;
		swap(x, x1);
		swap(a, b);
	}
	b = abs(q / d);
	x = (c / d) % b * (x % b) % b;
	if (x < 0) x += b;
	return {x, (ll)((c - (lll)p * x) / q)};
}
ll fun(ll a, ll b, ll p)//ax=b(mod p)
{
	return exgcd(a, -p, b).first % p;
}

