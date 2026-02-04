setwd("C:\\Users\\nahomw\\Desktop\\from_mac\\nahomworku\\Desktop\\uthealth\\gra_project\\proj_2")
library(readr)
library(dplyr)
df <- read_csv("FluSurveillance_Custom_Download_Data.csv", locale = locale(encoding = "UTF-8"), show_col_types = FALSE)

df <- df[,!(names(df) %in% c("CATCHMENT","NETWORK"))]
head(df)

corr_ <- function(columns){
  df_select <- df[,c(names(df),WEEKLY RATE) %in% c(columns)]
  df_clean <- df %>%
    filter(!columns %in% c("null", "Overall"))
}

columns = c("SEX CATEGORY")
df_select <- df[,c(names(df),"WEEKLY RATE") %in% c(columns)]

df_select <- df %>%
  select(all_of(columns), `WEEKLY RATE`,"WEEK","YEAR...4") %>%
  filter(if_all(everything(), ~ !.x %in% c("null", "Overall"))) %>% 
  mutate(`WEEKLY RATE` = as.numeric(`WEEKLY RATE`),
         WEEK = as.numeric(WEEK))  

### PLOTS

ggplot(df_select, aes(x = WEEK, y = `WEEKLY RATE`, color = SEX_CODE)) +
  geom_line(size = 1) +
  geom_point(size = 1.5, alpha = 0.7) +
  facet_wrap(~ YEAR, scales = "free_y") +  # one subplot per year
  labs(title = "Weekly Rate Trends by Sex per Year",
       x = "Week of Year",
       y = "Weekly Rate",
       color = "Sex") +
  theme_minimal() +
  theme(strip.text = element_text(face = "bold"))





#########################################

# Data
Y <- c(22,24,25,24,29,35,24,32,33)
X <- c(5,4,6,2,6,6,4,7,8)
n <- length(Y)


# -------------------------

XTXinv <- c(n,sum(X),sum(X), sum(X^2))
XTXinv <- matrix(XTXinv, nrow = 2, ncol = 2)
XTXinv
aa <- c(1,5)
t(aa) %*% XTXinv %*% aa
# ------------------------------------

# Fit model
fit <- lm(Y ~ X)
coef_hat <- coef(fit)
s2 <- sum(resid(fit)^2)/(n-2)

# Design matrix
Xmat <- model.matrix(fit)  # includes intercept
XtX_inv <- solve(t(Xmat) %*% Xmat)

# Linear combination: alpha0*beta0 + alpha1*beta1
alpha0 <- 1
alpha1 <- 5
c_val <- 25

theta_hat <- alpha0*coef_hat[1] + alpha1*coef_hat[2]
SE_theta <- sqrt(s2 * c(alpha0, alpha1) %*% XtX_inv %*% c(alpha0, alpha1))

T_stat <- (theta_hat - c_val)/SE_theta
p_val <- 2*(1 - pt(abs(T_stat), df=n-2))

T_stat
p_val



