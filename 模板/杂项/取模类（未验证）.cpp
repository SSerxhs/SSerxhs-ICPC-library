struct Z
{
	ll x;
	Z():x(0){}
	Z(const ll &v):x(v){}
	static ll ksm(ll x,int y)
	{
		ll r=1;
		while (y)
		{
			if (y&1) r=r*x%p;
			x=x*x%p;y>>=1;
		}
		return r;
	}
	Z operator+(const Z &o) const {return {x+o.x>=p?x+o.x-p:x+o.x};}
	Z operator-(const Z &o) const {return {x<o.x?x-o.x+p:x-o.x};}
	Z operator*(const Z &o) const {return {x*o.x%p};}
	Z operator/(const Z &o) const {return {x*ksm(o.x,p-2)%p};}
	Z & operator+=(const Z &o) {if ((x+=o.x)>=p) x-=p;return *this;}
	Z & operator-=(const Z &o) {if ((x+=p-o.x)>=p) x-=p;return *this;}
	Z & operator*=(const Z &o) {x=x*o.x%p;return *this;}
	Z & operator/=(const Z &o) {x=x*ksm(o.x,p-2)%p;return *this;}
};
ostream & operator<<(ostream &cout,Z &v) {return cout<<v.x;}