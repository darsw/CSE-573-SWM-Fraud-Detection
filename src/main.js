var graph;
var innerHeight;
var innerWidth;
var male_data;
var female_data;
var defaultCountry = 'Estonia';
var completeData;
const margin = { top: 20, bottom: 90, right: 20, left: 160 };
var select
var data;
var selectedCustomer = 'C1093826151';

document.addEventListener('DOMContentLoaded', function () {
   

    
    select = d3.select('#customer-select').on('change', updateCharts);

    Promise.all([d3.csv('../data/modified_filtered_customers_data.csv'), (data) => {
        return {
            'customer': data.customer,
            'age': data.age,
            'amount': data.amount,
            'category': data.category,
            'fraud': data.fraud,
            'gender': data.gender,
            'merchant': data.merchant, 
            'step': data.step, 
            'zipMerchant': data.zipMerchant,
            'zipcodeOri': data.zipcodeOri,
            'transaction_type': data.transaction_type
        }
    }])
        .then(function (values) {
            console.log('loaded females_data.csv and males_data.csv');
          
            console.log(values[0])
            data = values[0];
            select.selectAll('option')
            .data(data)
            .enter()
            .append('option')
            .text(function(d) { return d.customer; });
        });
        updateCharts();
});

function updateCharts() {
    selectedCustomer = select.property('value');
    var customerData = data.filter(d => d.customer === selectedCustomer);
    const transactionCounts = countTransactionsByType(customerData);

    console.log(transactionCounts);
    transactionCounts['Travel and Transport']

    console.log(countTransactionsByType)
    transactionData = Object.entries(transactionCounts).map(d => ({ key: d[0], value: d[1] }));

    var fraudSums = calculateFraudSums(customerData);
    console.log(fraudSums);

    var fraudData = [
        { key: 'Legitimate', value: fraudSums.legitimate_sum },
        { key: 'Fraudulent', value: fraudSums.fraudulent_sum }
    ];

    createPieChart('#piechart', transactionData, 'Transaction Types');
    createPieChart('#fraudchart', fraudData, 'Fraudulent vs Legitimate Transactions');
}

function createPieChart(elementId, data, title) {
    // Dimensions and margins of the graph
    var width = 850, height = 400, margin = 40;
    var radius = Math.min(width, height) / 2 - margin;

    // Clear previous chart
    d3.select(elementId).html('');

    // Create SVG for pie chart
    var svg = d3.select(elementId)
      .append('svg')
        .attr('width', width)
        .attr('height', height)
      .append('g')
        .attr('transform', 'translate(' + width / 2 + ',' + height / 2 + ')');

    // Set color scale
    var color = d3.scaleOrdinal()
      .domain(data.map(d => d.key))
      .range(d3.schemeCategory10);

    // Create pie chart
    var pie = d3.pie().value(d => d.value);
    var path = d3.arc().outerRadius(radius).innerRadius(0);
    var labelArc = d3.arc().outerRadius(radius - 40).innerRadius(radius - 40); 


    var arcs = svg.selectAll('.arc')
        .data(pie(data))
        .enter()
        .append('g')
        .attr('class', 'arc');

    arcs.append('path')
        .attr('d', path)
        .attr('fill', d => color(d.data.key));


        var lastLabelEnd = -Infinity;
        var labelPadding = 3;
    
        var legendG = svg.selectAll(".legend")
        .data(pie(data))
        .enter().append("g")
        .attr("transform", function(d,i){
            return "translate(" + (radius + 20) + "," + (i * 15 - radius) + ")";
        })
        .attr("class", "legend");   

    legendG.append("rect")
        .attr("width", 10)
        .attr("height", 10)
        .attr("fill", d => color(d.data.key));

    legendG.append("text")
        .text(d => `${d.data.key}: ${d.data.value}`)
        .style("font-size", 12)
        .attr("y", 10)
        .attr("x", 11);
}

function changeCountry() {
    defaultCountry = document.getElementById("countries").value;
    console.log("the current country chosen is", defaultCountry);
    drawLolliPopChart()
}
const countTransactionsByType = (customerData) => {
    return customerData.reduce((acc, transaction) => {
        const type = transaction.transaction_type;

        if (!acc[type]) {
            acc[type] = 0;
        }
        acc[type]++;

        return acc;
    }, {});
};

function calculateFraudSums(customerData) {
    const sums = customerData.reduce((acc, transaction) => {
      
        if (transaction.fraud === 1) {
            acc.fraudulent_sum += 1;
        } else {
            acc.legitimate_sum += 1;
        }
        return acc;
    }, { fraudulent_sum: 0, legitimate_sum: 0 });

    return sums;
}


