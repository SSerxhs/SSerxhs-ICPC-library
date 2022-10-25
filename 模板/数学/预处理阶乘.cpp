namespace _fac
{
	typedef unsigned int ui;
	typedef unsigned long long ll;
	const int N=1e6+2;
	const ll p=998244353;
	ll fac[N];
	ll ifac[N];
	ll inv[N];
	ll ksm(ll x,int y)
	{
		ll r=1;
		while (y)
		{
			if (y&1) r=(ll)r*x%p;
			x=(ll)x*x%p;
			y>>=1;
		}
		return r;
	}
	ll C(int n,int m)
	{
		if (n<m||m<0) return 0;
		return (ll)fac[n]*ifac[m]%p*ifac[n-m]%p;
	}
	void init_fac()
	{
		int i;
		fac[0]=1;
		for (i=1; i<N; i++) fac[i]=(ll)fac[i-1]*i%p;

		ifac[N-1]=ksm(fac[N-1],p-2);
		for (i=N-1; i; i--) ifac[i-1]=(ll)ifac[i]*i%p;

		ll x; inv[1]=1;
		for (i=2; i<N; i++)
		{
			x=p/i;
			inv[i]=(ll)x*(p-inv[p-x*i])%p;
		}
	}
}
using _fac::init_fac; using _fac::fac; using _fac::ifac; using _fac::inv;
using _fac::ksm; using _fac::C; using _fac::p;
