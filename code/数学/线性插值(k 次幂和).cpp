ull interpolation(vector<ull> a, ull n)
{
	int m = a.size(), i;
	vector<ull> ans(2);
	n %= p;
	if (n < m) return a[n];
	ull k = ifac[m - 1];
	for (i = m - 1; i >= 0; i--)
	{
		(a[i] *= k) %= p;
		(k *= n - i) %= p;
	}
	k = 1;
	for (i = 0; i < m; i++)
	{
		(ans[(m ^ i) & 1] += a[i] * k) %= p;
		k = k * inv[i + 1] % p * (n - i) % p * (m - i - 1) % p;
	}
	return (ans[1] + p - ans[0]) % p;
}
ull sum_of_kth_power(ull n, ull k)
{
	if (n == 0) return 0;
	ull m = min(n + 1, k + 2);
	int i;
	vector<ull> s(m);
	vector<int> pr, ed(m); pr.reserve(m / 4);
	s[1] = 1;
	for (i = 2; i < m; i++)
	{
		if (!ed[i]) s[i] = ksm(i, k), pr.push_back(i);
		for (int j : pr) if (i * j < m)
		{
			s[i * j] = s[i] * s[j] % p;
			if (i % j == 0) break;
		}
		else break;
	}
	for (i = 1; i < m; i++) (s[i] += s[i - 1]) %= p;
	return interpolation(s, n);
}
