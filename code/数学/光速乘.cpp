ll mul(ll x, ll y)
{
	x = x * y - (ll)((ldb)x / p * y + 1e-8) * p;
	if (x < 0) return x + p; return x;
}
