
import numpy as np
import pandas as pd
import statistics as st 
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import anderson



df = pd.read_csv("netflix_cleaned.csv")
mean = np.mean(df['Rating'])

print("Ratings Ortalaması:", mean)


df = pd.read_csv("netflix_cleaned.csv")
numberofratingmean = np.mean(df['Number of Ratings'])

print("Değerlendirme Sayısı Ortalaması:", numberofratingmean)

df = pd.read_csv("netflix_cleaned.csv")
hoursviewedmean = np.mean(df['Hours Viewed'])

print("İzlenme Sayısı Ortalaması:",hoursviewedmean)



df = pd.read_csv("netflix_cleaned.csv")

mode = st.mode(df['Rating'])
print("Ratings'in modu:", mode)

df = pd.read_csv("netflix_cleaned.csv")

numberofratingmode = st.mode(df['Number of Ratings'])
print("Değerlendirme Sayısı modu:", numberofratingmode)

df = pd.read_csv("netflix_cleaned.csv")

hoursviewedmode = st.mode(df['Hours Viewed'])
print("Ortalama izlenme Sayısı modu:", hoursviewedmode)


df = pd.read_csv("netflix_cleaned.csv")

median=df[['Rating']].median()
print("Rating medyanı:",median)

df = pd.read_csv("netflix_cleaned.csv")

numberofratingsmedian=df[['Number of Ratings']].median()
print("Değerlendirme Sayısı medyanı:",numberofratingsmedian)


df = pd.read_csv("netflix_cleaned.csv")

hoursviewmedian=df[['Hours Viewed']].median()
print("Ortalama izlenme sayısı medyanı:",hoursviewedmean)


df = pd.read_csv("netflix_cleaned.csv")
print(df.describe()) #tanımlayıcı istatistikler



df = pd.read_csv("netflix_cleaned.csv")
variance = st.variance(df.Rating)
print("Rating varyansı:",variance)

df = pd.read_csv("netflix_cleaned.csv")
numberofratingvariance = df[['Number of Ratings']].var()
print("Değerlendirme Sayısı Varyansı:",numberofratingvariance)

df = pd.read_csv("netflix_cleaned.csv")
hoursviewedvariance = df[['Hours Viewed']].var()
print("Ortalama İzlenme Sayısı Varyansı:",hoursviewedmean)



df = pd.read_csv("netflix_cleaned.csv")
ss=st.stdev(df.Rating)
print("Rating Standart Sapması:",ss)

df = pd.read_csv("netflix_cleaned.csv")
numberofratingss=df[['Number of Ratings']].std()
print("Değerlendirme Sayısı Standart Sapması:",numberofratingss)

df = pd.read_csv("netflix_cleaned.csv")
hoursviewss= df[['Hours Viewed']].std()
print("Ortalama İzlenme Sayısı Standart Sapması:",hoursviewss)



df = pd.read_csv("netflix_cleaned.csv")
ptp=np.ptp(df.Rating)
print("En yüksek ve en düşük Rating Farkı:",ptp)





df = pd.read_csv("netflix_cleaned.csv")


print("Number of Ratings Sütunu Önce:")
print(df['Number of Ratings'])


non_numeric_values = df[~df['Number of Ratings'].astype(str).str.isnumeric()]['Number of Ratings']


try:
    df['Number of Ratings'] = pd.to_numeric(df['Number of Ratings'], errors='coerce')
    df = df.dropna(subset=['Number of Ratings'])
except Exception as e:
    print("Hata:", e)


print("\nNumber of Ratings Sütunu Sonra:")
print(df['Number of Ratings'])

df = pd.read_csv("netflix_cleaned.csv")
numberofratingptp = np.ptp(df['Number of Ratings'])
print("En yüksek ve en düşük Değerlendirme Sayısı Farkı:", numberofratingptp)

hoursviewptp = np.ptp(df['Hours Viewed'])
print("En yüksek ve en düşük İzlenme Sayısı Farkı:", hoursviewptp)

if numberofratingptp >100000 :
    print("Değerlendirme Sayılarında Gözle Görülür Fark Söz Konusu")
else:
    print ("Değerlendirme Sayılarında Gözle Görülür Fark Söz Konusu Değildir")



df = pd.read_csv("netflix_cleaned.csv")
print("Eksik veya Hatalı Değerler:")
print(df['Rating'].isnull().sum())  
print(df['Rating'].dtype)  



skewness1 = skew(df['Rating'], bias=False)
print("Rating çarpıklık:", skewness1)


if -0.5 < skewness1 < 0.5:
    print("Simetrik , Rating değerlerinin dağılımı eşittir")
elif skewness1 < 0:
    print("Sola Çarpık")
elif skewness1 > 0:
    print("Sağa Çarpık")


df = pd.read_csv("netflix_cleaned.csv")  
skewness2 = skew(df['Number of Ratings'], bias=False)
print("Number of Ratings çarpıklık:", skewness2)


if -0.5 < skewness2 < 0.5:
    print("Simetrik")
elif skewness2 < 0:
    print("Sola Çarpık")
elif skewness2 > 0:
    print("Sağa Çarpık") 


df = pd.read_csv("netflix_cleaned.csv")  
skewness3 = skew(df['Hours Viewed'], bias=False)
print("Hours Viewed çarpıklık:", skewness3)


if -0.5 < skewness3 < 0.5:
    print("Simetrik")
elif skewness3 < 0:
    print("Sola Çarpık")
elif skewness3 > 0:
    print("Sağa Çarpık")   

df = pd.read_csv("netflix_cleaned.csv")
rating_kurtosis = df['Rating'].kurtosis()
print("Rating sütunu basıklık:", rating_kurtosis)

if -1<rating_kurtosis<1:
    print("normal dağılım")
elif rating_kurtosis>0:
    print("sivri dağılım")
elif rating_kurtosis<0:
    print("basık dağılım")        


df = pd.read_csv("netflix_cleaned.csv")
number_of_ratings_kurtosis = df['Number of Ratings'].kurtosis()
print("Number of Ratings sütunu çarpıklık:", number_of_ratings_kurtosis)

if -1 < number_of_ratings_kurtosis < 1:
    print("Normal dağılım")
elif number_of_ratings_kurtosis > 0:
    print("Sivri dağılım")
elif number_of_ratings_kurtosis < 0:
    print("Basık dağılım")    


df = pd.read_csv("netflix_cleaned.csv")
hours_viewed_kurtosis = df['Hours Viewed'].kurtosis()
print("Hours Viewed sütunu çarpıklık:", hours_viewed_kurtosis)


if -1 < hours_viewed_kurtosis < 1:
    print("Normal dağılım")
elif hours_viewed_kurtosis > 0:
    print("Sivri dağılım")
elif hours_viewed_kurtosis < 0:
    print("Basık dağılım")    


print("Rating Sütununun Çeyreklik Değerleri :")
rating_quantiles=st.quantiles(df.Rating,n=5) 
print("İlk çeyreklik (%25):", rating_quantiles[0])
print("Medyan (%50):", rating_quantiles[1])
print("Üçüncü çeyreklik (%75):", rating_quantiles[2])
print("Son çeyreklik (%100):", rating_quantiles[3])


print("Number of Rating Sütununun Çeyreklik Değerleri :")
df = pd.read_csv("netflix_cleaned.csv")
number_of_ratings_quantiles = st.quantiles(df['Number of Ratings'],n=5)

print("İlk çeyreklik (%25):", number_of_ratings_quantiles[0])
print("Medyan (%50):", number_of_ratings_quantiles[1])
print("Üçüncü çeyreklik (%75):", number_of_ratings_quantiles[2])
print("Son çeyreklik (%100):", number_of_ratings_quantiles[3])


print("Hours View Sütununun Çeyreklik Değerleri :")

df = pd.read_csv("netflix_cleaned.csv")
hours_view_quantiles = st.quantiles(df['Hours Viewed'],n=5)

print("İlk çeyreklik (%25):", hours_view_quantiles[0])
print("Medyan (%50):", hours_view_quantiles[1])
print("Üçüncü çeyreklik (%75):", hours_view_quantiles[2])
print("Son çeyreklik (%100):", hours_view_quantiles[3])


df = pd.read_csv("netflix_cleaned.csv")

rating_column = 'Rating'
ratings = df[rating_column]

# Histogram çizimi
ratings.plot.hist(bins=10, edgecolor='black')
plt.title('Rating Histogram')
plt.xlabel('Rating Değerleri')
plt.ylabel('Frekans')
plt.show()


df = pd.read_csv("netflix_cleaned.csv")  

number_of_ratings_column = 'Number of Ratings'
number_of_ratings = df[number_of_ratings_column]

# Seaborn ile histogram çizimi
sns.histplot(number_of_ratings, bins=5, kde=True, color='skyblue', edgecolor='black')
plt.title('Number of Ratings Histogram')
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.show()

df = pd.read_csv("netflix_cleaned.csv")  

rating_column = 'Rating'
rating_data = df[rating_column]

# Shapiro-Wilk testi
stat, p_value = shapiro(rating_data)


alpha = 0.05
print(f'Statistic: {stat}, p-value: {p_value}')

if p_value > alpha:
    print("Veri normal dağılıma uyar (H0 reddedilemez)")
else:
    print("Veri normal dağılıma uymaz (H0 reddedilir)")



# "Number of Ratings" sütununu seç
number_of_ratings_column = 'Number of Ratings'
number_of_ratings_data = df[number_of_ratings_column]

# Anderson-Darling testi
result = anderson(number_of_ratings_data)

# İstatistik ve elemanlar
statistic = result.statistic
critical_values = result.critical_values
significance_level = result.significance_level

# H0 hipotezi: Veri normal dağılıma uyar
print(f'Anderson-Darling İstatistik: {statistic}')

# Elemanlar ve karşılık gelen elemanlar arasındaki karşılaştırma
for i in range(len(critical_values)):
    print(f'Eleman {i + 1}: {critical_values[i]}, Anlam düzeyi: {significance_level[i]}')

# H0 hipotezi reddedilecekse
if statistic > critical_values[2]:
    print("Number of Ratings verisi normal dağılıma uymaz (H0 reddedilir)")
else:
    print("Number of Ratings verisi normal dağılıma uyar (H0 reddedilemez)")



df = pd.read_csv("netflix_cleaned.csv")  

sns.boxplot(x=df["Rating"])
Rating =df["Rating"]

aykiri=np.where(Rating>10)
print("Aykırı Değerlerin İndeksleri:", aykiri)

if len(aykiri) == 0:
    print("Aykırı değer yoktur")
else:
    print("Aykırı değer vardır")






































































