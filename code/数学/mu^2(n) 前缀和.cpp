const int N = 5e7 + 5;
int pr[N / 8], cnt, mu[N];
bool ed[N];
void init()
{
	ui i, j, k;
	mu[1] = 1;
	for (i = 2; i < N; i++)
	{
		if (!ed[i]) pr[++cnt] = i, mu[i] = -1;
		for (j = 1; pr[j] * i < N; j++)
		{
			ed[pr[j] * i] = 1;
			if (i % pr[j] == 0) break;
			mu[pr[j] * i] = -mu[i];
		}
		mu[i] += mu[i - 1];
	}
}
ll sum_mu(ll n)
{
	if (n < N) return mu[n];
	ll r = 1, i, j, k;
	for (i = 2; i <= n; i = j + 1)
	{
		j = n / (k = n / i);
		r -= sum_mu(k) * (j - i + 1);
	}
	return r;
}
ll sum_mu2(ll n)
{
	ll r = 0, i, j, k, l, s = 0, t;
	for (i = 1; i * i <= n; i = j + 1)
	{
		k = n / (i * i);
		j = sqrtl(n / k);
		t = sum_mu(j);
		r += k * (t - s);
		s = t;
	}
	return r;
}
int main()
{
	ll n;
	init();
	cin >> n;
	cout << sum_mu2(n) << endl;
}
