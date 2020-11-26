clc
clear
%% read data
M = csvread('Cancer.csv');
Ms = normalize(M(:, 3:end));
% normalized data with label in the 1st column
Ms = [M(:, 2) Ms];
%% PCA
X = Ms(:, 2:end);
[coeff,~,~,~,explained] = pca(X);
% reduced new data that explains more than 99% of variance
trX = X * coeff(:, 1:17);
%% visualization
% not normalized
Xr = tsne(M(:, 3:end));
% normalized
Xrr = tsne(X);
% after pca
Xpr = tsne(trX);
figure
scatter(Xr(i0, 1), Xr(i0, 2), 'filled', 'r')
hold on
scatter(Xr(i1, 1), Xr(i1, 2), 'filled', 'b')
title('Visualization before Normalization')
xlabel('Feature I')
ylabel('Feature II')
legend('benign (class 0)', 'malignant (class 1)')
hold off
figure
scatter(Xrr(i0, 1), Xrr(i0, 2), 'filled', 'r')
hold on
scatter(Xrr(i1, 1), Xrr(i1, 2), 'filled', 'b')
title('Visualization after Normalization')
xlabel('Feature I')
ylabel('Feature II')
legend('benign (class 0)', 'malignant (class 1)')
hold off
figure
scatter(Xpr(i0, 1), Xpr(i0, 2), 'filled', 'r')
hold on
scatter(Xpr(i1, 1), Xpr(i1, 2), 'filled', 'b')
title('Visualization after PCA')
xlabel('Feature I')
ylabel('Feature II')
legend('benign (class 0)', 'malignant (class 1)')
hold off
%% change label for estimating beta for logit
NY = Ms(:, 1);
NY(NY == 1) = 2;
NY(NY == 0) = 1;
%% forward feature selection
s_r = num2cell(1:17);
rest = trX;
s_g = {};
get = [];
s_s = {};
select = [];
top = 0;
beta = sum(i0) / sum(i1);
while ~isempty(rest)
    score = zeros(1, size(rest, 2));
    for i = 1:size(rest, 2)
       x = [get rest(:, i)];
       label = Bayeslmd(x, mean(x(i0, :)), cov(x(i0, :)), mean(x(i1, :)), cov(x(i1, :)), 1, beta);
%        label = KNNlmd([Ms(:, 1) x], x, 18, 1, 0.5);
%        Mdl = fitcsvm(x, Ms(:, 1), 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
%        Mdl = fitcsvm(x, Ms(:, 1), 'Solver', 'SMO');
%        label = predict(Mdl,x);
%        label = LDAlmd([Ms(:, 1) x], x, 1, 0.0020);
%        score(i) = 1 - mean(Ms(:, 1) ~= label);
%        label = Logitlmd([NY x], x, 1, 0.6213);
       score(i) = 1 - mean(NY ~= label); 
    end
    [M, I] = max(score);
    if M > top
        top = M;
        s_s = [s_g s_r(I)];
        select = [get rest(:, I)];
    end
    s_g = [s_g s_r(I)];
    get = [get rest(:, I)];
    s_r(I) = [];
    rest(:, I) = [];
end
% display PCs that score the most for classification
disp(s_s)
%% Bayes
Xb = tsne([trX(:, 1:6) trX(:, 8:10)]);
r0 = Xb(Ms(:, 1) == 0, :);
r1 = Xb(Ms(:, 1) == 1, :);
beta = size(r0, 1) / size(r1, 1);
desurfB(r0, r1, mean(r0), cov(r0), mean(r1), cov(r1), beta);
title('Gaussian Bayes Decision Boundary after Applying t-SNE')
hold off
%% SVM Linear/Gaussian
% linear
Xs1 = tsne([trX(:, 1:10) trX(:, 13:14) trX(:, 16)]);
figure
desurfS([Ms(:, 1) Xs1], 0.5, 0);
title('SVM with Linear Kernel Decision Boundary after Applying t-SNE')
hold off
% gaussian
Xs2 = tsne([trX(:, 1:4) trX(:, 6:9) trX(:, 11)]);
figure
desurfS([Ms(:, 1) Xs2], 0.5, 1);
title('SVM with Gaussian Kernel Decision Boundary after Applying t-SNE')
hold off
%% finding best threshold for LDA/Logit based on accuracy
% best thre
bestt = 0;
% highest acc
score = 0;
% Current fold thre
cur = 0;
% current fold highest score
sc = 0;
cv2 = cvpartition(569, 'k', 10);
for l = 1:10
for j = 1:cv2.NumTestSets
    tri = cv2.training(j);
    tei = cv2.test(j);
    i0 = Ms(tei, 1) == 0;
    i1 = Ms(tei, 1) == 1;
%     lmd = LDAlmd([Ms(tri, 1) trX(tri, :)], trX(tei, :), 0, 0);
    lmd = Logitlmd([NY(tri) trX(tri, :)], trX(tei, :), 0, 0);
    thre = [-Inf; lmd; Inf];
    for i = 1:size(thre, 1)
%         lb = LDAlmd([Ms(tri, 1) trX(tri, :)], trX(tei, :), 1, thre(i));
%         sc = 1 - mean(Ms(tei, 1) ~= lb);
        lb = Logitlmd([NY(tri) trX(tri, :)], trX(tei, :), 1, thre(i));
        sc = 1 - mean(NY(tei) ~= lb);
        if sc > score
            score = sc;
            cur = thre(i);
        end
    end
    bestt = bestt + cur;
end
end
bestt = bestt / 100;
fprintf('The best threshold is %.4f\n', bestt);
%% Logit
Xlogit = tsne([trX(:, 1:5) trX(:, 7:10) trX(:, 12) trX(:, 14) trX(:, 16) trX(:, 17)]);
desurfLogit([NY Xlogit], 0.6213);
title('Logistic Decision Boundary after Applying t-SNE')
hold off
%% LDA
Xl = tsne(trX(:, 1:8));
desurfL([Ms(:, 1) Xl], 0.0020);
title('LDA Decision Boundary after Applying t-SNE')
hold off
%% KNN find best k
for l = 1:10
cv1 = cvpartition(569, 'k', 10);
kk = floor(linspace(1, 560));
Y = zeros(size(kk, 2), 1);
X = 569 ./ kk;
for j = 1:cv1.NumTestSets
    tri = cv1.training(j);
    tei = cv1.test(j);
    i0 = Ms(tei, 1) == 0;
    i1 = Ms(tei, 1) == 1;
    for k = 1:size(kk, 2)
        thre = [genthre(kk(k)); Inf];
        lmd = KNNlmd([Ms(tri, 1) trX(tri, :)], trX(tei, :), kk(k), 0, 0);
        [roc] = genROC(lmd(i0), lmd(i1), thre);
        pcd = calper(roc.Pfa, roc.Pd, p0, 0);
        Y(k) = Y(k) + (1 - pcd);
    end
end
end
Y = Y / 100;
plot(X, Y)
xlabel('N / k')
ylabel('min Pe')
title('KNN Bias and Flexibility trade-off')
legend('10-fold CV')
%% KNN best threshold
thre = genthre(18);
cv = cvpartition(569, 'k', 10);
% best thre
bes = 0;
% highest acc
score = 0;
% Current fold thre
cur = 0;
for l = 1:10
for j = 1:cv.NumTestSets
    tri = cv.training(j);
    tei = cv.test(j);
    for i = 1:size(thre, 1)
        lmd = KNNlmd([Ms(tri, 1) trX(tri, :)], trX(tei, :), 18, 1, thre(i));
        sc = 1 - mean(Ms(tei, 1) ~= lmd);
        if sc > score
            score = sc;
            cur = thre(i);
        end
    end
    bes = bes + cur;
end
end
bes = bes / 100;
fprintf('The best threshold for KNN is %.4f\n', bes);
%% KNN
% for 30 features
Xk = tsne([trX(:, 1:2) trX(:, 4:5) trX(:, 8) trX(:, 10) trX(:, 12) trX(:, 14:15)]);
desurfK([Ms(:, 1) Xk], 18, 0.5);
% previous for 10 features
% Xk = tsne([trX(:, 1:2) trX(:, 4:5) trX(:, 9:10) trX(:, 12)]);
title('KNN Decision Boundary after Applying t-SNE at k = 18')
hold off
%% six classifiers together
Db = [Ms(:, 1) trX(:, 1:6) trX(:, 8:10)];
Dk = [Ms(:, 1) trX(:, 1:2) trX(:, 4:5) trX(:, 8) trX(:, 10) trX(:, 12) trX(:, 14:15)];
Dsl = [Ms(:, 1) trX(:, 1:10) trX(:, 13:14) trX(:, 16)];
Dsg = [Ms(:, 1) trX(:, 1:4) trX(:, 6:9) trX(:, 11)];
Dl = [Ms(:, 1) trX(:, 1:8)];
Dlogit = [NY trX(:, 1:5) trX(:, 7:10) trX(:, 12) trX(:, 14) trX(:, 16) trX(:, 17)];

[b, k, sl, sg, l, logit] = Classify(Db, Dk, Dsl, Dsg, Dl, Dlogit);
fprintf('Accuracy for Gaussian Bayes is: %.4f\n', b)
fprintf('Accuracy for KNN is: %.4f\n', k)
fprintf('Accuracy for Linear SVM is: %.4f\n', sl)
fprintf('Accuracy for Gaussian SVM is: %.4f\n', sg)
fprintf('Accuracy for LDA is: %.4f\n', l)
fprintf('Accuracy for Logistic Regression is: %.4f\n', logit)

CVRoc(Db, Dk, Dsl, Dsg, Dl, Dlogit);
hold off
%% Functions

function lmd = Bayeslmd(te, m0, s0, m1, s1, lbon, beta)
% lbon = 1 for switching to label
% beta for threshold
lmd = zeros(size(te, 1), 1);
for i = 1:size(te, 1)
   y1 = mvnpdf(te(i, :), m1, s1);
   y0 = mvnpdf(te(i, :), m0, s0);
   lmd(i) = log(y1 / y0);
   if lbon == 1
       if lmd(i) >= log(beta)
           lmd(i) = 1;
       else
           lmd(i) = 0;
       end
   end
end
end
function desurfB(r0, r1, m0, s0, m1, s1, z)
D = [r0; r1];
xR = max(D(:, 1)) - min(D(:, 1));
yR = max(D(:, 2)) - min(D(:, 2));
a = linspace(min(D(:, 1)) - 0.2 * xR, max(D(:, 1)) + 0.2 * xR, 251);
b = linspace(min(D(:, 2)) - 0.2 * yR, max(D(:, 2)) + 0.2 * yR, 251);
[X, Y] = meshgrid(a, b);
Xl = reshape(X, [], 1);
Yl = reshape(Y, [], 1);
lmd = Bayeslmd([Xl Yl], m0, s0, m1, s1, 0, 0);
imagesc(a([1 end]),b([1 end]), reshape(lmd, 251, 251));
colorbar
hold on
set(gca,'YDir','normal');
v = [z z];
contour(X, Y, reshape(lmd, 251, 251), v, 'm', 'LineWidth', 3)
scatter(r0(:, 1), r0(:, 2), 'r', 'filled')
scatter(r1(:, 1), r1(:, 2), 'b', 'filled')
ylabel('Feature II')
xlabel('Feature I')
legend('decision boundary', 'class 0', 'class 1')
end

function lmd = KNNlmd(tr, te, k, lbon, beta)
% lbon = 1 for switching to label
% beta for threshold
lmd = zeros(size(te, 1), 1);
for i = 1:size(te, 1)
   Idx = knnsearch(tr(:, 2:end), te(i, :), 'K', k, 'Distance', 'Euclidean');
   count = 0;
   % count how many belongs to H1
   for k = 1:size(Idx, 2)
       if tr(Idx(k), 1) == 1
          count = count + 1; 
       end
   end
   lmd(i) = count / k;
   if lbon == 1
       if lmd(i) >= beta
           lmd(i) = 1;
       else
           lmd(i) = 0;
       end
   end
end
end
function desurfK(D, k, z)
xR = max(D(:, 2)) - min(D(:, 2));
yR = max(D(:, 3)) - min(D(:, 3));
a = linspace(min(D(:, 2)) - 0.2 * xR, max(D(:, 2)) + 0.2 * xR, 251);
b = linspace(min(D(:, 3)) - 0.2 * yR, max(D(:, 3)) + 0.2 * yR, 251);
[X, Y] = meshgrid(a, b);
Xl = reshape(X, [], 1);
Yl = reshape(Y, [], 1);
lmd = KNNlmd(D, [Xl Yl], k, 0, 0);
imagesc(a([1 end]),b([1 end]), reshape(lmd, 251, 251));
colorbar
hold on
set(gca,'YDir','normal');
v = [z z];
contour(X, Y, reshape(lmd, 251, 251), v, 'm', 'LineWidth', 3)
scatter(D(D(:, 1) == 0, 2), D(D(:, 1) == 0, 3), 'r', 'filled')
scatter(D(D(:, 1) == 1, 2), D(D(:, 1) == 1, 3), 'b', 'filled')
ylabel('Feature II')
xlabel('Feature I')
legend('decision boundary', 'class 0', 'class 1')
end

function lmd = LDAlmd(tr, te, lbon, beta)
% lbon = 1 for switching to label
% beta for threshold
G0 = tr(tr(:, 1) == 0, 2:end);
G1 = tr(tr(:, 1) == 1, 2:end);
m0 = sum(G0)' / size(G0, 1);
m1 = sum(G1)' / size(G1, 1);
s0 = (G0' - m0) * (G0' - m0)';
s1 = (G1' - m1) * (G1' - m1)';
sw = s0 + s1;
% data demean
lmd = ((sw \ (m1 - m0))' * (te - mean(te))')';
if lbon == 1
    lmd(lmd >= beta) = 1;
    lmd(lmd < beta) = 0;
end
end
function desurfL(D, z)
xR = max(D(:, 2)) - min(D(:, 2));
yR = max(D(:, 3)) - min(D(:, 3));
a = linspace(min(D(:, 2)) - 0.2 * xR, max(D(:, 2)) + 0.2 * xR, 251);
b = linspace(min(D(:, 3)) - 0.2 * yR, max(D(:, 3)) + 0.2 * yR, 251);
[X, Y] = meshgrid(a, b);
Xl = reshape(X, [], 1);
Yl = reshape(Y, [], 1);
lmd = LDAlmd(D, [Xl Yl], 0, 0);
imagesc(a([1 end]),b([1 end]), reshape(lmd, 251, 251));
colorbar
hold on
set(gca,'YDir','normal');
v = [z z];
contour(X, Y, reshape(lmd, 251, 251), v, 'm', 'LineWidth', 3)
scatter(D(D(:, 1) == 0, 2), D(D(:, 1) == 0, 3), 'r', 'filled')
scatter(D(D(:, 1) == 1, 2), D(D(:, 1) == 1, 3), 'b', 'filled')
ylabel('Feature II')
xlabel('Feature I')
legend('decision boundary', 'class 0', 'class 1')
end

function lmd = Logitlmd(tr, te, lbon, beta)
% lbon = 1 for switching to label
% beta for threshold
x = tr(:, 2:end);
y = tr(:, 1);
b = mnrfit(x, y);
lmd = zeros(size(te, 1), 1);
for i = 1:size(te, 1)
   lmd(i) = 1 / (1 + exp(1) ^ (b' * [1; te(i, :)']));
   if lbon == 1
       if lmd(i) >= beta
           lmd(i) = 2;
       else
           lmd(i) = 1;
       end
   end
end
end
function desurfLogit(D, z)
xR = max(D(:, 2)) - min(D(:, 2));
yR = max(D(:, 3)) - min(D(:, 3));
a = linspace(min(D(:, 2)) - 0.2 * xR, max(D(:, 2)) + 0.2 * xR, 251);
b = linspace(min(D(:, 3)) - 0.2 * yR, max(D(:, 3)) + 0.2 * yR, 251);
[X, Y] = meshgrid(a, b);
Xl = reshape(X, [], 1);
Yl = reshape(Y, [], 1);
lmd = Logitlmd(D, [Xl Yl], 0, 0);
imagesc(a([1 end]),b([1 end]), reshape(lmd, 251, 251));
colorbar
hold on
set(gca,'YDir','normal');
v = [z z];
contour(X, Y, reshape(lmd, 251, 251), v, 'm', 'LineWidth', 3)
scatter(D(D(:, 1) == 1, 2), D(D(:, 1) == 1, 3), 'r', 'filled')
scatter(D(D(:, 1) == 2, 2), D(D(:, 1) == 2, 3), 'b', 'filled')
ylabel('Feature II')
xlabel('Feature I')
legend('decision boundary', 'class 0', 'class 1')
end

function desurfS(D, z, opt)
% opt = 1 for gaussian kernel
xR = max(D(:, 2)) - min(D(:, 2));
yR = max(D(:, 3)) - min(D(:, 3));
a = linspace(min(D(:, 2)) - 0.2 * xR, max(D(:, 2)) + 0.2 * xR, 251);
b = linspace(min(D(:, 3)) - 0.2 * yR, max(D(:, 3)) + 0.2 * yR, 251);
[X, Y] = meshgrid(a, b);
Xl = reshape(X, [], 1);
Yl = reshape(Y, [], 1);
if opt == 1
    mdl = fitcsvm(D(:, 2:end), D(:, 1), 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
else
    mdl = fitcsvm(D(:, 2:end), D(:, 1), 'Solver', 'SMO');
end
mdl = fitPosterior(mdl);
[~, score] = predict(mdl, [Xl Yl]);
lmd = score(:, 2);
imagesc(a([1 end]),b([1 end]), reshape(lmd, 251, 251));
colorbar
hold on
set(gca,'YDir','normal');
v = [z z];
contour(X, Y, reshape(lmd, 251, 251), v, 'm', 'LineWidth', 3)
scatter(D(D(:, 1) == 0, 2), D(D(:, 1) == 0, 3), 'r', 'filled')
scatter(D(D(:, 1) == 1, 2), D(D(:, 1) == 1, 3), 'b', 'filled')
ylabel('Feature II')
xlabel('Feature I')
legend('decision boundary', 'class 0', 'class 1')
end

% performance
function [b, k, sl, sg, l, logit] = Classify(Db, Dk, Dsl, Dsg, Dl, Dlogit)
cv = cvpartition(size(Db, 1), 'k', 10);
[be, ke, sle, sge, le, logite] = deal(0);
for j = 1: cv.NumTestSets
    tri = cv.training(j);
    tei = cv.test(j);
    % DO NOT MESS UP WITH INDICES!!!
    Tr = Db(tri, :);
    r0 = Tr(Tr(:, 1) == 0, 2:end);
    r1 = Tr(Tr(:, 1) == 1, 2:end);
    blmd = Bayeslmd(Db(tei, 2:end), mean(r0), cov(r0), mean(r1), cov(r1), 1, size(r0, 1) / size(r1, 1));
    klmd = KNNlmd(Dk(tri, :), Dk(tei, 2:end), 18, 1, 0.5);
    llmd = LDAlmd(Dl(tri, :), Dl(tei, 2:end), 1, 0.0020);
    logitlmd = Logitlmd(Dlogit(tri, :), Dlogit(tei, 2:end), 1, 0.6213);
    % linear kernel
    Mdll = fitcsvm(Dsl(tri, 2:end), Dsl(tri, 1), 'Solver', 'SMO');
    sllmd = predict(Mdll,Dsl(tei, 2:end));
    % gaussian kernel
    Mdlg = fitcsvm(Dsg(tri, 2:end), Dsg(tri, 1), 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
    sglmd = predict(Mdlg,Dsg(tei, 2:end));
    
    be = be + mean(Db(tei, 1) ~= blmd);
    ke = ke + mean(Dk(tei, 1) ~= klmd);
    sle = sle + mean(Dsl(tei, 1) ~= sllmd);
    sge = sge + mean(Dsg(tei, 1) ~= sglmd);
    le = le + mean(Dl(tei, 1) ~= llmd);
    logite = logite + mean(Dlogit(tei, 1) ~= logitlmd);
end
b = 1 - be/10;
k = 1 - ke/10;
sl = 1 - sle/10;
sg = 1 - sge/10;
l = 1 - le/10;
logit = 1 - logite/10;
end
function CVRoc(Db, Dk, Dsl, Dsg, Dl, Dlogit)
cv = cvpartition(size(Db, 1), 'k', 10);
[b0, b1, k0, k1, l0, l1, logit0, logit1] = deal([]);
[suml, sumg, gatherl, gatherg] = deal([]);
for j = 1: cv.NumTestSets
    tri = cv.training(j);
    tei = cv.test(j);
    i0 = Db(tei, 1) == 0;
    i1 = Db(tei, 1) == 1;
    % DO NOT MESS UP WITH INDICES!!!
    Tr = Db(tri, :);
    r0 = Tr(Tr(:, 1) == 0, 2:end);
    r1 = Tr(Tr(:, 1) == 1, 2:end);
    blmd = Bayeslmd(Db(tei, 2:end), mean(r0), cov(r0), mean(r1), cov(r1), 0, 0);
    
    klmd = KNNlmd(Dk(tri, :), Dk(tei, 2:end), 18, 0, 0);
    llmd = LDAlmd(Dl(tri, :), Dl(tei, 2:end), 0, 0);
    logitlmd = Logitlmd(Dlogit(tri, :), Dlogit(tei, 2:end), 0, 0);
    % linear kernel
    slmdl = fitcsvm(Dsl(tri, 2:end), Dsl(tri, 1), 'Solver', 'SMO');
    lmdl = fitPosterior(slmdl);
    [~, scorel] = predict(lmdl, Dsl(tei, 2:end));
    suml = [suml; scorel(:, 2)];
    gatherl = [gatherl; Dsl(tei, 1)];
    % gaussian kernel
    sgmdl = fitcsvm(Dsg(tri, 2:end), Dsg(tri, 1), 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
    gmdl = fitPosterior(sgmdl);
    [~, scoreg] = predict(gmdl, Dsg(tei, 2:end));
    sumg = [sumg; scoreg(:, 2)];
    gatherg = [gatherg; Dsg(tei, 1)];
    
    b0 = [b0; blmd(i0)];
    b1 = [b1; blmd(i1)];
    k0 = [k0; klmd(i0)];
    k1 = [k1; klmd(i1)];
    l0 = [l0; llmd(i0)];
    l1 = [l1; llmd(i1)];
    logit0 = [logit0; logitlmd(i0)];
    logit1 = [logit1; logitlmd(i1)];
end
[rocb] = genROC(b0, b1, sort([b0; b1]));
threk = [-Inf; genthre(18); Inf];
[rock] = genROC(k0, k1, threk);
[rocl] = genROC(l0, l1, sort([l0; l1]));
[roclogit] = genROC(logit0, logit1, sort([logit0; logit1]));
% for SVM
[xl,yl,~,cl] = perfcurve(gatherl, suml, 1);
[xg,yg,~,cg] = perfcurve(gatherg, sumg, 1);

[p1, ~] = calper(rocb.Pfa, rocb.Pd, 0.6274, 1);
[p2, ~] = calper(rock.Pfa, rock.Pd, 0.6274, 1);
[p3, ~] = calper(xl, yl, 0.6274, 1);
[p4, ~] = calper(xg, yg, 0.6274, 1);
[p5, ~] = calper(rocl.Pfa, rocl.Pd, 0.6274, 1);
[p6, ~] = calper(roclogit.Pfa, roclogit.Pd, 0.6274, 1);

title('10-fold CV ROCs with maxPcds')
legend('Gaussian Bayes', 'KNN', 'Linear SVM', 'Gaussian SVM', 'LDA', 'Logitstic Regression')
xlabel('Pfa')
ylabel('Pd')
fprintf('Gaussian Bayes maxPcd: %.4f%%, AUC: %.4f%%\n', p1, -1*trapz(rocb.Pfa, rocb.Pd));
fprintf('KNN maxPcd: %.4f%%, AUC: %.4f%%\n', p2, -1*trapz(rock.Pfa, rock.Pd));
fprintf('Linear SVM maxPcd: %.4f%%, AUC: %.4f%%\n', p3, cl);
fprintf('Gaussian SVM maxPcd: %.4f%%, AUC: %.4f%%\n', p4, cg);
fprintf('LDA maxPcd: %.4f%%, AUC: %.4f%%\n', p5, -1*trapz(rocl.Pfa, rocl.Pd));
fprintf('Logistic Regression maxPcd: %.4f%%, AUC: %.4f%%\n', p6, -1*trapz(roclogit.Pfa, roclogit.Pd));
end

function thre = genthre(k)
thre = zeros(k+1, 1);
for i = 2:(k+1)
    thre(i) = (i - 1) / k;
end
end
function [roc] = genROC(null, alt, thre)
sz0 = size(null, 1);
sz1 = size(alt, 1);
sz2 = size(thre, 1);
roc.Pfa = zeros(sz2, 1);
roc.Pd = zeros(sz2, 1);
for i = 1:sz2
    count0 = 0;
    count1 = 0;
    for j = 1:sz0
        if null(j) >= thre(i)
            count0 = count0 + 1;
        end
    end
    for k = 1:sz1
       if alt(k) >= thre(i)
            count1 = count1 + 1;
       end 
    end
    roc.Pfa(i) = count0 / sz0;
    roc.Pd(i) = count1 / sz1;
end
end
function [mxPcd, idx] = calper(pfa, pd, pH0, star)
% star = 1 for marking maxPcd
cmp = 0;
for i = 1:size(pfa, 1)
    if (1 - pfa(i)) * pH0 + pd(i) * (1 - pH0) > cmp
        cmp = (1 - pfa(i)) * pH0 + pd(i) * (1 - pH0);
        idx = i;
    end
end
mxPcd = cmp;
if star == 1
    plot(pfa, pd, '-*', 'MarkerIndices', idx)
    hold on
end
end
