% Author:  <ashik@KAI10>
% Created: 2017-03-19

clear

Train = dlmread('trainLinearlySeparable.txt');
% disp(Train);

Header = Train(1,:);

numberOfFeatures = Header(1,1)
numberOfClass = Header(1,2)
numberOfExample = Header(1,3)

Class = Train(2:end, numberOfFeatures+1);
Train = Train(2:end, 1:numberOfFeatures);

% [numberOfExample numberOfFeatures] = size(Train)

W = rand(numberOfFeatures+1, 1);
rho = 0.1;
cycle = 0;

while(1)
    cycle = cycle + 1;
  
    % update W if needed 
    for i=1:numberOfExample
        X = [Train(i,:)'; 1];
        val = W' * X;
        if(Class(i) == 1 && val <= 0) W = W + rho*X;
        elseif(Class(i) == 2 && val >= 0) W = W - rho*X;
        end
    end
    
    % check if any example is misclassified
    done = 1; 
    for i=1:numberOfExample
        X = [Train(i,:)'; 1];
        val = W' * X;
        if((Class(i) == 1 && val <= 0) || (Class(i) == 2 && val >=0))
            done = 0;
            break;
        end
    end
    
    if(done) break;
    end
end

fprintf('NO. of cycles: %d\n', cycle);
fprintf('Result W vector:\n');
disp(W);

% ############################################################################

fprintf('\n*************** Train data processing complete ****************\n');
fprintf('***************************************************************\n');

for i=1:numberOfFeatures
    fprintf('Feature %d\t', i);
end

fprintf('actual class\tpredicted class\n');
fprintf('---------------------------------------------------------------\n');
for i=1:numberOfExample
    val = W' * [Train(i,:)'; 1];
    predict = 1;
    if(val < 0) predict = 2;
    end;
    
    for j=1:numberOfFeatures
        fprintf('%f\t', Train(i,j));
    end

    fprintf('%8d%16d\n', Class(i), predict);
end

fprintf('***************************************************************\n\n');

% ############################################################################

% starting Test

Test = dlmread('testLinearlySeparable.txt');
testClass = Test(:, numberOfFeatures+1);
Test = Test(:,1:numberOfFeatures);
[testExamples testFeatures] = size(Test)

correct = 0;
true_positive = 0;
false_positive = 0;
true_negative = 0;
false_negative = 0;

for i=1:testExamples
    % fprintf('correct: %d\n', correct);
    X = [Test(i,:)'; 1];
    val = W' * X;
    if(val > 0 && testClass(i) == 1) true_positive = true_positive + 1;
    elseif(val > 0 && testClass(i) == 2) false_positive = false_positive + 1;
    elseif(val < 0 && testClass(i) == 2) true_negative = true_negative + 1;
    elseif(val < 0 && testClass(i) == 1) false_negative = false_negative + 1;
    end
 
end

condition_positive = true_positive + false_negative;
condition_negative = false_positive + true_negative;

test_outcome_positive = true_positive + false_positive;
test_outcome_negative = false_negative + true_negative;

fprintf('\naccuracy: %f\n', 100.0*(true_positive+true_positive)/testExamples);
fprintf('precision: %f\n', 100.0*true_positive/test_outcome_positive);
fprintf('recall: %f\n', 100.0*true_positive/condition_positive);
